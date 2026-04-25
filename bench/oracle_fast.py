"""
Fast oracle for LAVE false-negative detection.

What is actually expensive vs. not
----------------------------------
* **Prefix replay** is handled by incremental sync only: ``OracleState.sync_to_prefix``
  advances the oracle's ``Checker`` from the last synced length to ``len(prefix)``.
  That cost is **O(len(prefix)) per distinct prefix**, not per DFS branch.

* **Search** (DFS/BFS) runs only over the **block** tokens (e.g. length 32), not the
  full prompt+prefix. Each branch uses ``consume`` / ``rollback`` (DFS) or
  ``clone_state`` (BFS) on top of the matcher state **after** the prefix — i.e.
  the same ``n`` as ``block_len`` in complexity intuition, not ``prefix+block``.

* **Snapshot API**: llguidance exposes ``LLMatcher.deep_copy()`` (wrapped here as
  ``Checker.clone_state()``). There is no serialize/deserialize; BFS clones in RAM.

Search modes (sound vs heuristic)
---------------------------------
* **dfs** (default): one ``Checker``, rollback, early exit — **sound** w.r.t. the
  matcher, smallest memory. Recommended for FN detection and benchmarking.

* **bfs**: full breadth-first with a clone per branch — **sound**, but memory can
  explode; use to cross-check ``dfs``.

* **bfs_dedup**: merges frontier states that share a **heuristic key** (see
  ``_dedup_state_key``). Same next-token mask does **not** imply the same internal
  grammar state in general CFGs; merging can drop valid assignments (false
  negatives vs ``dfs``). A warning is emitted once per process when this mode is
  selected. Use only for performance experiments; compare against ``dfs`` on hard
  schemas (e.g. ``anyOf``).

* **smart** (``DGRAMMAR_ORACLE_SEARCH_MODE=smart``): phase-1 ``bfs_dedup``; if it
  returns ``True``, accept immediately (fast SAT — **can** be wrong if dedup has a
  false positive). If phase-1 returns ``False``, phase-2 **sound** ``dfs`` confirms
  unsat or finds a satisfying path missed by dedup. TN cost is roughly
  ``dedup_time + dfs_time``; SAT-often workloads can win when phase-1 hits quickly.
  Use ``bench/oracle_compare_dedup_dfs.py`` to estimate dedup vs DFS disagreement.

* **Richer dedup fingerprint (optional)**: env ``DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS`` —
  comma-separated token ids (e.g. JSON punctuation / structural ids for LLaDA). For
  each id, append ``compute_logit_bias()`` bytes after a one-token ``try_consume`` on
  a **matcher clone**, reducing accidental merges in ``bfs_dedup`` / ``smart`` phase-1.
  Still not provably sound; measure with the comparison script.

* **Per-call search time cap (optional)**: env ``DGRAMMAR_ORACLE_MAX_SEARCH_SECONDS=S`` or
  ``oracle_verify_fast(..., max_search_seconds=S)``. Applies **after** prefix sync; if DFS/BFS
  exceeds ``S`` seconds, the call returns ``None`` with ``skip_reason=oracle_search_timeout``.
  Does not bound ``sync_to_prefix`` time. Use to avoid wall-clock blowups on heavy instances.

* **Trie ordering (optional, default off)**: env ``DGRAMMAR_ORACLE_TRIE_ORDER=1`` builds a
  **prefix trie** over single-token debug strings (``LLTokenizer.dbg_tokens``) for the
  current ``allowed_ids``, then tries token ids in **trie DFS (preorder)** — grouping
  candidates by shared string prefixes (helps JSON string–heavy TN exploration order).
  With logits priority on, **logits dominate**; trie order breaks ties. Does not remove
  branches (still sound).

* **Logits priority (optional, default on)**: when the FN-detection patch passes
  ``model_logits`` + ``logits_index_base`` (same ``p`` tensor as ``validate()``), MASK
  branches try allowed token ids in **descending model score** at that sequence position.
  This matches LAVE's beam ordering heuristic and typically speeds up **FN** searches
  (valid path exists but default id order explored slowly). Does not change soundness of
  DFS/BFS. Disable with env ``DGRAMMAR_ORACLE_LOGITS_PRIORITY=0``.

* **String-style branching (optional)**: env ``DGRAMMAR_ORACLE_STRING_PRUNE_THRESHOLD=N``
  (or ``oracle_verify_fast(..., string_prune_threshold=N)``). At each MASK, if
  ``compute_mask`` yields **more than N** allowed token ids (after dropping MASK),
  only the **first** allowed id is tried. This targets JSON **string** positions
  where hundreds of character tokens are legal — it is **not** sound: the oracle
  may miss a valid completion and change FN/TN counts vs full enumeration. Default
  is off (no threshold). Use for TN-heavy slow runs; compare against full DFS when
  reporting false-negative rates.

Optimizations implemented
-------------------------
* Grammar + tokenizer cache: one ``Checker`` per instance (``OracleState``).
* Incremental prefix sync (tail consume / rollback).
* DFS with **empty-mask prune** and **fixed-suffix fast path** (no MASK left → linear
  consume + single rollback).

**Token-class grouping** (merge allowed ids into grammar-equivalence classes) is not
implemented: that would require llguidance internals beyond ``compute_mask``.

**Iterative deepening** on mask count does not change semantics when every MASK
position must be filled; it is not enabled here.

Must use ``model_name=\"LLaDA\"`` so tokenization matches the generation ``Checker``.
"""

from __future__ import annotations

import os
import time
import warnings
from collections import deque
from typing import Any, Dict, Literal, Optional, Tuple

mask_id = 126336

# (id(LLTokenizer), token_id) -> dbg string for trie keys (cleared on excessive growth).
_TOKEN_TEXT_CACHE: dict[tuple[int, int], str] = {}
_TOKEN_TEXT_CACHE_MAX = 400_000


class _TrieNode:
    __slots__ = ("children", "leaf_ids")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.leaf_ids: list[int] = []


def _token_text_for_trie(checker: Any, tid: int) -> str:
    """Single-token string for trie paths; uses ``LLTokenizer.dbg_tokens``."""
    if tid == mask_id:
        return ""
    tok = checker.tokenizer
    key = (id(tok), tid)
    if key in _TOKEN_TEXT_CACHE:
        return _TOKEN_TEXT_CACHE[key]
    try:
        s = tok.dbg_tokens([tid])
        if not isinstance(s, str):
            s = str(s)
    except Exception:
        s = f"\uffff<{tid}>"
    if len(_TOKEN_TEXT_CACHE) >= _TOKEN_TEXT_CACHE_MAX:
        _TOKEN_TEXT_CACHE.clear()
    _TOKEN_TEXT_CACHE[key] = s
    return s


def _trie_insert(root: _TrieNode, text: str, tid: int) -> None:
    if not text:
        root.leaf_ids.append(tid)
        return
    node = root
    for ch in text:
        if ch not in node.children:
            node.children[ch] = _TrieNode()
        node = node.children[ch]
    node.leaf_ids.append(tid)


def _trie_dfs_collect(node: _TrieNode, out: list[int]) -> None:
    out.extend(sorted(node.leaf_ids))
    for ch in sorted(node.children.keys()):
        _trie_dfs_collect(node.children[ch], out)


def _trie_dfs_order(allowed_ids: list[int], checker: Any) -> list[int]:
    """Deterministic order: trie preorder over UTF-32 code point edges."""
    root = _TrieNode()
    for tid in allowed_ids:
        if tid == mask_id:
            continue
        txt = _token_text_for_trie(checker, tid)
        _trie_insert(root, txt, tid)
    out: list[int] = []
    _trie_dfs_collect(root, out)
    return out


def _resolve_trie_order(explicit: Optional[bool]) -> bool:
    """Order MASK candidates by prefix trie over token dbg strings. Default off."""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("DGRAMMAR_ORACLE_TRIE_ORDER", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return False

SearchMode = Literal["dfs", "bfs", "bfs_dedup", "smart"]


class OracleSearchTimeout(Exception):
    """Raised when block search exceeds ``max_search_seconds`` (cooperative deadline)."""


def _check_search_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise OracleSearchTimeout()


def _resolve_logits_priority(explicit: Optional[bool]) -> bool:
    """Prefer trying high-``p`` tokens at MASK when logits are provided. Default True."""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("DGRAMMAR_ORACLE_LOGITS_PRIORITY", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


def _sort_allowed_ids_by_logits(
    allowed_ids: list[int],
    logits_row: Any,
) -> list[int]:
    """``logits_row`` is a 1-D tensor over vocab (e.g. ``p[0, pos]``)."""
    scored: list[tuple[float, int]] = []
    for tid in allowed_ids:
        if tid == mask_id:
            continue
        try:
            v = float(logits_row[tid].item())
        except Exception:
            v = 0.0
        scored.append((v, tid))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [tid for _, tid in scored]


def _order_mask_branch_candidates(
    allowed_ids: list[int],
    pos_in_block: int,
    checker: Any,
    logits_pair: Optional[Tuple[Any, int]],
    prioritize_logits: bool,
    trie_priority: bool,
) -> list[int]:
    """
    Reorder MASK branch candidates: optional ``trie_priority`` (prefix trie over
    ``dbg_tokens``), optional ``prioritize_logits``. If both, sort by descending logit
    then trie preorder rank as tiebreak.
    """
    if not allowed_ids:
        return allowed_ids

    mask_tail = [t for t in allowed_ids if t == mask_id]
    ids = [t for t in allowed_ids if t != mask_id]
    if not ids:
        return allowed_ids

    has_logits = prioritize_logits and logits_pair is not None
    row = None
    if has_logits:
        p, base = logits_pair  # type: ignore[misc]
        g = base + pos_in_block
        if 0 <= g < p.shape[1]:
            row = p[0, g]

    if not trie_priority:
        if row is not None:
            return _sort_allowed_ids_by_logits(ids, row) + mask_tail
        return allowed_ids

    trie_order_list = _trie_dfs_order(ids, checker)
    trie_rank = {tid: i for i, tid in enumerate(trie_order_list)}

    if row is not None:
        def sort_key(tid: int) -> tuple[float, int]:
            try:
                lv = float(row[tid].item())
            except Exception:
                lv = 0.0
            return (-lv, trie_rank.get(tid, 10**9))

        return sorted(ids, key=sort_key) + mask_tail

    return trie_order_list + mask_tail


def _resolve_max_search_seconds(explicit: Optional[float]) -> Optional[float]:
    """Positive float = cap; None or <=0 = no cap (env ``DGRAMMAR_ORACLE_MAX_SEARCH_SECONDS``)."""
    if explicit is not None:
        return explicit if explicit > 0 else None
    raw = os.environ.get("DGRAMMAR_ORACLE_MAX_SEARCH_SECONDS", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        return None


class OracleState:
    """One ``Checker`` per instance; prefix advanced incrementally."""

    def __init__(self, grammar_str: str, model_name: str = "LLaDA") -> None:
        from constrained_diffusion.checker_tokenizer import Checker

        self.checker = Checker(grammar_str, model_name=model_name)
        self._synced_len = 0

    def sync_to_prefix(self, prefix_tokens: list[int]) -> bool:
        """Advance checker to match ``prefix_tokens``; only consume new tail."""
        target_len = len(prefix_tokens)

        if target_len < self._synced_len:
            self.checker.rollback(self._synced_len - target_len)
            self._synced_len = target_len
            return True

        if target_len == self._synced_len:
            return True

        new_tokens = prefix_tokens[self._synced_len : target_len]
        ok = self.checker.consume_tokens(new_tokens)
        if ok:
            self._synced_len = target_len
        return bool(ok)


def _default_search_mode() -> SearchMode:
    v = os.environ.get("DGRAMMAR_ORACLE_SEARCH_MODE", "dfs").strip().lower()
    if v in ("bfs", "bfs_dedup", "dfs", "smart"):
        return v  # type: ignore[return-value]
    return "dfs"


_DedupKey = Tuple[
    bytes,
    bool,
    bool,
    str,
    int,
    Tuple[bytes, ...],
]


def _resolve_dedup_probe_token_ids() -> tuple[int, ...]:
    """Optional comma-separated token ids (env) to refine ``_dedup_state_key``."""
    raw = os.environ.get("DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS", "").strip()
    if not raw:
        return ()
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(out)


def _dedup_probe_bias_after_token(m0: Any, tid: int) -> bytes:
    """
    One-step lookahead bias for dedup keys: clone matcher, try one token, return
    post-state mask bytes (or a short sentinel).
    """
    if tid == mask_id:
        return b"!"
    m = m0.deep_copy()
    if m.is_stopped() or m.is_error():
        return b"S"
    cnt = m.try_consume_tokens([tid])
    if cnt != 1:
        if cnt:
            m.rollback(cnt)
        return b"X"
    if m.is_error():
        return b"E"
    try:
        return bytes(m.compute_logit_bias())
    except Exception:
        return b"?"


def _dedup_probe_tuple(m0: Any, probe_tids: tuple[int, ...]) -> Tuple[bytes, ...]:
    if not probe_tids:
        return ()
    return tuple(_dedup_probe_bias_after_token(m0, tid) for tid in probe_tids)


def _dedup_state_key(checker) -> _DedupKey:
    """
    Heuristic merge key for ``bfs_dedup``. **Not** a canonical grammar-state ID:
    two matcher states can share this key and still diverge later. Includes more
    than ``compute_logit_bias`` alone (accepting / stopped / stop_reason / capture
    hash) to reduce accidental merges vs mask-only.
    """
    m = checker.matcher
    if m.is_error():
        return (b"", True, False, "error", 0, ())
    bias = bytes(m.compute_logit_bias())
    sr = m.stop_reason()
    sr_s = sr if isinstance(sr, str) else str(sr)
    caps = m.get_captures()
    if caps:
        cap_h = hash(tuple((name, bytes(val)) for name, val in caps))
    else:
        cap_h = 0
    probes = _dedup_probe_tuple(m, _resolve_dedup_probe_token_ids())
    return (
        bias,
        bool(m.is_accepting()),
        bool(m.is_stopped()),
        sr_s,
        cap_h,
        probes,
    )


_BFS_DEDUP_WARNED = False
_STRING_PRUNE_WARNED = False
_SMART_WARNED = False


def _string_prune_threshold_resolved(explicit: Optional[int]) -> Optional[int]:
    """Positive int = enable; None = off."""
    if explicit is not None:
        return explicit if explicit > 0 else None
    raw = os.environ.get("DGRAMMAR_ORACLE_STRING_PRUNE_THRESHOLD", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return None


def _warn_string_prune_once(threshold: int) -> None:
    global _STRING_PRUNE_WARNED
    if _STRING_PRUNE_WARNED:
        return
    _STRING_PRUNE_WARNED = True
    warnings.warn(
        f"oracle string_prune_threshold={threshold}: at MASK positions with more "
        f"than {threshold} allowed tokens, only one token is tried — oracle is not "
        f"sound vs full enumeration.",
        UserWarning,
        stacklevel=2,
    )


def _prune_mask_allowed_ids(
    allowed_ids: list[int],
    threshold: Optional[int],
) -> list[int]:
    """
    If branching factor exceeds ``threshold``, keep a single representative
    (first non-MASK id). Otherwise return all non-MASK ids.
    """
    ids = [t for t in allowed_ids if t != mask_id]
    if not ids:
        return []
    if threshold is None or len(ids) <= threshold:
        return ids
    return [ids[0]]


def _warn_bfs_dedup_once() -> None:
    global _BFS_DEDUP_WARNED
    if _BFS_DEDUP_WARNED:
        return
    _BFS_DEDUP_WARNED = True
    warnings.warn(
        "oracle search_mode=bfs_dedup merges branches using a heuristic key; "
        "distinct llguidance states can collide → results may disagree with dfs/bfs. "
        "Use dfs (default) or bfs for a sound oracle.",
        UserWarning,
        stacklevel=2,
    )


def _warn_smart_once() -> None:
    global _SMART_WARNED
    if _SMART_WARNED:
        return
    _SMART_WARNED = True
    warnings.warn(
        "oracle search_mode=smart: if phase-1 bfs_dedup returns True, phase-2 DFS is "
        "skipped (fast SAT). A dedup false-positive could make the oracle wrong; "
        "phase-1 False is always confirmed with sound DFS. See bench/oracle_compare_dedup_dfs.py.",
        UserWarning,
        stacklevel=2,
    )


def _suffix_mask_counts(block_tokens: list[int]) -> list[int]:
    """masks_suffix[i] = number of mask_id in block_tokens[i:]."""
    n = len(block_tokens)
    out = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        out[i] = out[i + 1] + (1 if block_tokens[i] == mask_id else 0)
    return out


def _oracle_dfs(
    checker,
    block_tokens: list[int],
    pos: int,
    masks_suffix: list[int],
    string_prune_threshold: Optional[int],
    deadline: Optional[float] = None,
    logits_pair: Optional[Tuple[Any, int]] = None,
    prioritize_logits: bool = False,
    trie_priority: bool = False,
) -> bool:
    _check_search_deadline(deadline)
    n = len(block_tokens)
    if pos == n:
        return True

    # No MASK left: single deterministic path (no branching).
    if masks_suffix[pos] == 0:
        consumed = 0
        i = pos
        while i < n:
            _check_search_deadline(deadline)
            tok = block_tokens[i]
            if checker.is_stoped():
                if consumed:
                    checker.rollback(consumed)
                return False
            if not checker.validate_tokens([tok]):
                if consumed:
                    checker.rollback(consumed)
                return False
            checker.consume_tokens([tok])
            consumed += 1
            i += 1
        if consumed:
            checker.rollback(consumed)
        return True

    tok = block_tokens[pos]

    if tok != mask_id:
        if checker.is_stoped():
            return False
        if not checker.validate_tokens([tok]):
            return False
        checker.consume_tokens([tok])
        ok = _oracle_dfs(
            checker,
            block_tokens,
            pos + 1,
            masks_suffix,
            string_prune_threshold,
            deadline,
            logits_pair,
            prioritize_logits,
            trie_priority,
        )
        checker.rollback(1)
        return ok

    if checker.is_stoped():
        return False

    # Remaining slots must cover remaining MASKs (sanity prune; usually tautology).
    slots_left = n - pos
    if masks_suffix[pos] > slots_left:
        return False

    allowed_mask = checker.compute_mask()
    allowed_ids = allowed_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(allowed_ids, int):
        allowed_ids = [allowed_ids]
    allowed_ids = _prune_mask_allowed_ids(allowed_ids, string_prune_threshold)
    if not allowed_ids:
        return False

    allowed_ids = _order_mask_branch_candidates(
        allowed_ids,
        pos,
        checker,
        logits_pair,
        prioritize_logits,
        trie_priority,
    )

    for tid in allowed_ids:
        if tid == mask_id:
            continue
        checker.consume_tokens([tid])
        if _oracle_dfs(
            checker,
            block_tokens,
            pos + 1,
            masks_suffix,
            string_prune_threshold,
            deadline,
            logits_pair,
            prioritize_logits,
            trie_priority,
        ):
            checker.rollback(1)
            return True
        checker.rollback(1)
    return False


def _oracle_dfs_assignment(
    checker,
    block_tokens: list[int],
    pos: int,
    masks_suffix: list[int],
    string_prune_threshold: Optional[int],
    deadline: Optional[float] = None,
    logits_pair: Optional[Tuple[Any, int]] = None,
    prioritize_logits: bool = False,
    trie_priority: bool = False,
) -> Optional[list[int]]:
    """
    Same search as ``_oracle_dfs``, but returns the **concrete token sequence** for the
    whole block ``block_tokens[0:n]`` when satisfiable; ``None`` if unsat.
    """
    _check_search_deadline(deadline)
    n = len(block_tokens)
    if pos == n:
        return []

    if masks_suffix[pos] == 0:
        segment: list[int] = []
        consumed = 0
        i = pos
        while i < n:
            _check_search_deadline(deadline)
            tok = block_tokens[i]
            if checker.is_stoped():
                if consumed:
                    checker.rollback(consumed)
                return None
            if not checker.validate_tokens([tok]):
                if consumed:
                    checker.rollback(consumed)
                return None
            checker.consume_tokens([tok])
            segment.append(tok)
            consumed += 1
            i += 1
        if consumed:
            checker.rollback(consumed)
        return segment

    tok = block_tokens[pos]

    if tok != mask_id:
        if checker.is_stoped():
            return None
        if not checker.validate_tokens([tok]):
            return None
        checker.consume_tokens([tok])
        rest = _oracle_dfs_assignment(
            checker,
            block_tokens,
            pos + 1,
            masks_suffix,
            string_prune_threshold,
            deadline,
            logits_pair,
            prioritize_logits,
            trie_priority,
        )
        checker.rollback(1)
        if rest is None:
            return None
        return [tok] + rest

    if checker.is_stoped():
        return None

    slots_left = n - pos
    if masks_suffix[pos] > slots_left:
        return None

    allowed_mask = checker.compute_mask()
    allowed_ids = allowed_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(allowed_ids, int):
        allowed_ids = [allowed_ids]
    allowed_ids = _prune_mask_allowed_ids(allowed_ids, string_prune_threshold)
    if not allowed_ids:
        return None

    allowed_ids = _order_mask_branch_candidates(
        allowed_ids,
        pos,
        checker,
        logits_pair,
        prioritize_logits,
        trie_priority,
    )

    for tid in allowed_ids:
        if tid == mask_id:
            continue
        checker.consume_tokens([tid])
        rest = _oracle_dfs_assignment(
            checker,
            block_tokens,
            pos + 1,
            masks_suffix,
            string_prune_threshold,
            deadline,
            logits_pair,
            prioritize_logits,
            trie_priority,
        )
        checker.rollback(1)
        if rest is not None:
            return [tid] + rest
    return None


def _oracle_bfs(
    checker,
    block_tokens: list[int],
    masks_suffix: list[int],
    string_prune_threshold: Optional[int],
    deadline: Optional[float] = None,
    logits_pair: Optional[Tuple[Any, int]] = None,
    prioritize_logits: bool = False,
    trie_priority: bool = False,
) -> bool:
    """Breadth-first search with explicit ``Checker.clone_state`` at MASK branches."""
    n = len(block_tokens)
    q: deque[tuple[Any, int]] = deque([(checker.clone_state(), 0)])

    while q:
        _check_search_deadline(deadline)
        c, pos = q.popleft()
        if pos == n:
            return True

        if masks_suffix[pos] == 0:
            consumed = 0
            i = pos
            while i < n:
                tok = block_tokens[i]
                if c.is_stoped():
                    break
                if not c.validate_tokens([tok]):
                    break
                c.consume_tokens([tok])
                consumed += 1
                i += 1
            if i == n:
                return True
            continue

        tok = block_tokens[pos]
        if tok != mask_id:
            if c.is_stoped():
                continue
            if not c.validate_tokens([tok]):
                continue
            c.consume_tokens([tok])
            q.append((c, pos + 1))
            continue

        if c.is_stoped():
            continue
        if masks_suffix[pos] > n - pos:
            continue

        allowed_mask = c.compute_mask()
        allowed_ids = allowed_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(allowed_ids, int):
            allowed_ids = [allowed_ids]
        allowed_ids = _prune_mask_allowed_ids(allowed_ids, string_prune_threshold)
        if not allowed_ids:
            continue

        allowed_ids = _order_mask_branch_candidates(
            allowed_ids,
            pos,
            c,
            logits_pair,
            prioritize_logits,
            trie_priority,
        )

        for tid in allowed_ids:
            if tid == mask_id:
                continue
            nc = c.clone_state()
            if not nc.consume_tokens([tid]):
                continue
            q.append((nc, pos + 1))

    return False


def _finish_block_fixed_tokens(checker, block_tokens: list[int], pos: int, n: int) -> bool:
    """Consume block_tokens[pos:n] (must contain no MASK). Mutates ``checker``."""
    i = pos
    while i < n:
        tok = block_tokens[i]
        if checker.is_stoped():
            return False
        if not checker.validate_tokens([tok]):
            return False
        checker.consume_tokens([tok])
        i += 1
    return True


def _oracle_bfs_dedup(
    checker,
    block_tokens: list[int],
    masks_suffix: list[int],
    string_prune_threshold: Optional[int],
    deadline: Optional[float] = None,
    logits_pair: Optional[Tuple[Any, int]] = None,
    prioritize_logits: bool = False,
    trie_priority: bool = False,
) -> bool:
    """
    Breadth-first expansion with **heuristic** merging: one representative per
    ``(_dedup_state_key(checker_after_step), masks_used)``. Still not sound in
    general — see module docstring.
    """
    n = len(block_tokens)
    total_masks = masks_suffix[0]
    ch0 = checker.clone_state()
    layer: dict[tuple[_DedupKey, int], Any] = {
        (_dedup_state_key(ch0), 0): ch0,
    }

    pos = 0
    while pos < n:
        _check_search_deadline(deadline)
        if not layer:
            return False

        if masks_suffix[pos] == 0:
            # Suffix has no MASK: every viable path must already have used all MASKs.
            for (_dk, mu), ch in list(layer.items()):
                if mu != total_masks:
                    continue
                cc = ch.clone_state()
                if _finish_block_fixed_tokens(cc, block_tokens, pos, n):
                    return True
            return False

        next_layer: dict[tuple[_DedupKey, int], Any] = {}
        for _bi, ((_, mu0), ch) in enumerate(layer.items()):
            if _bi % 64 == 0:
                _check_search_deadline(deadline)
            tok = block_tokens[pos]

            if tok != mask_id:
                if ch.is_stoped():
                    continue
                if not ch.validate_tokens([tok]):
                    continue
                nc = ch.clone_state()
                if not nc.consume_tokens([tok]):
                    continue
                fp2 = _dedup_state_key(nc)
                nk = (fp2, mu0)
                if nk not in next_layer:
                    next_layer[nk] = nc
                continue

            if ch.is_stoped():
                continue
            if masks_suffix[pos] > n - pos:
                continue

            allowed_mask = ch.compute_mask()
            allowed_ids = allowed_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(allowed_ids, int):
                allowed_ids = [allowed_ids]
            allowed_ids = _prune_mask_allowed_ids(allowed_ids, string_prune_threshold)
            if not allowed_ids:
                continue

            allowed_ids = _order_mask_branch_candidates(
                allowed_ids,
                pos,
                ch,
                logits_pair,
                prioritize_logits,
                trie_priority,
            )

            for tid in allowed_ids:
                if tid == mask_id:
                    continue
                nc = ch.clone_state()
                if not nc.consume_tokens([tid]):
                    continue
                fp2 = _dedup_state_key(nc)
                nk = (fp2, mu0 + 1)
                if nk not in next_layer:
                    next_layer[nk] = nc

        layer = next_layer
        pos += 1

    # Require that wildcard budget matches the block's MASK count (defensive;
    # sound paths should satisfy this; unsound dedup could leave spurious mu).
    return any(mu == total_masks for (_, mu) in layer.keys())


def _oracle_smart_verify(
    checker,
    block_tokens: list[int],
    masks_suffix: list[int],
    string_prune_threshold: Optional[int],
    deadline: Optional[float] = None,
    logits_pair: Optional[Tuple[Any, int]] = None,
    prioritize_logits: bool = False,
    trie_priority: bool = False,
) -> bool:
    """
    Phase-1 ``bfs_dedup`` (heuristic). If True, return True immediately.
    If False, phase-2 sound ``_oracle_dfs`` (may recover from dedup false negatives).
    """
    dedup_ok = _oracle_bfs_dedup(
        checker,
        block_tokens,
        masks_suffix,
        string_prune_threshold,
        deadline,
        logits_pair,
        prioritize_logits,
        trie_priority,
    )
    if dedup_ok:
        return True
    return _oracle_dfs(
        checker,
        block_tokens,
        0,
        masks_suffix,
        string_prune_threshold,
        deadline,
        logits_pair,
        prioritize_logits,
        trie_priority,
    )


def oracle_verify_fast(
    oracle_state: OracleState,
    prefix_tokens: list[int],
    block_tokens: list[int],
    oracle_mask_limit: int,
    *,
    search_mode: Optional[SearchMode] = None,
    timing_out: Optional[Dict[str, Any]] = None,
    string_prune_threshold: Optional[int] = None,
    max_search_seconds: Optional[float] = None,
    model_logits: Optional[Any] = None,
    logits_index_base: Optional[int] = None,
    logits_priority: Optional[bool] = None,
    trie_order: Optional[bool] = None,
) -> Optional[bool]:
    """
    If ``timing_out`` is a dict, it is filled with (milliseconds):

    - ``prefix_sync_ms`` — ``sync_to_prefix`` only (incremental replay vs last oracle call)
    - ``search_ms`` — DFS/BFS over the block only
    - ``total_ms`` — wall time for this call
    - ``search_mode`` — resolved mode string
    - ``n_masks`` — when the search phase ran
    - ``logits_priority`` / ``logits_priority_active`` — see module docstring
    - ``trie_order`` / ``trie_order_active`` — prefix trie over ``dbg_tokens`` (see ``_resolve_trie_order``)

    ``max_search_seconds`` caps **block search** time (after prefix sync). When exceeded,
    returns ``None`` with ``skip_reason="oracle_search_timeout"``. Resolved via
    ``_resolve_max_search_seconds`` (explicit arg, else env ``DGRAMMAR_ORACLE_MAX_SEARCH_SECONDS``).

    ``model_logits`` (``p`` from ``validate``, shape ``[1, L, vocab]``) and ``logits_index_base``
    (``index_to_consume``) enable MASK branch ordering by score; see ``_resolve_logits_priority``.

    On skip or failure, partial keys may still be set (see ``skip_reason``, ``sync_failed``, etc.).
    """
    t0 = time.perf_counter()
    mode = search_mode if search_mode is not None else _default_search_mode()
    lp = _resolve_logits_priority(logits_priority)
    tr = _resolve_trie_order(trie_order)
    str_prune = _string_prune_threshold_resolved(string_prune_threshold)
    if str_prune is not None:
        _warn_string_prune_once(str_prune)

    def _emit_timing(
        *,
        prefix_ms: float,
        search_ms: float,
        **extra: Any,
    ) -> None:
        if timing_out is None:
            return
        timing_out.clear()
        timing_out.update(
            {
                "prefix_sync_ms": round(prefix_ms, 3),
                "search_ms": round(search_ms, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000, 3),
                "search_mode": mode,
            }
        )
        timing_out.update(extra)

    n_masks = sum(1 for t in block_tokens if t == mask_id)
    if n_masks > oracle_mask_limit:
        _emit_timing(
            prefix_ms=0.0,
            search_ms=0.0,
            skipped=True,
            skip_reason="oracle_mask_limit",
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=False,
            trie_order=tr,
            trie_order_active=False,
        )
        return None

    if mode == "bfs_dedup":
        _warn_bfs_dedup_once()
    elif mode == "smart":
        _warn_smart_once()
        _warn_bfs_dedup_once()

    t_sync0 = time.perf_counter()
    sync_ok = oracle_state.sync_to_prefix(prefix_tokens)
    prefix_ms = (time.perf_counter() - t_sync0) * 1000
    if not sync_ok:
        _emit_timing(
            prefix_ms=prefix_ms,
            search_ms=0.0,
            sync_failed=True,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=False,
            trie_order=tr,
            trie_order_active=False,
        )
        return None

    prefix_len = len(oracle_state.checker.tokens)
    masks_suffix = _suffix_mask_counts(block_tokens)

    logits_pair: Optional[Tuple[Any, int]] = None
    if model_logits is not None and logits_index_base is not None:
        logits_pair = (model_logits, int(logits_index_base))
    logits_active = bool(lp and logits_pair is not None)

    max_s = _resolve_max_search_seconds(max_search_seconds)
    deadline = (time.perf_counter() + max_s) if max_s is not None else None

    t_s0 = time.perf_counter()
    smart_extra: Dict[str, Any] = {}
    try:
        if mode == "bfs":
            result = _oracle_bfs(
                oracle_state.checker,
                block_tokens,
                masks_suffix,
                str_prune,
                deadline,
                logits_pair,
                lp,
                tr,
            )
        elif mode == "bfs_dedup":
            result = _oracle_bfs_dedup(
                oracle_state.checker,
                block_tokens,
                masks_suffix,
                str_prune,
                deadline,
                logits_pair,
                lp,
                tr,
            )
        elif mode == "smart":
            t_d0 = time.perf_counter()
            dedup_ok = _oracle_bfs_dedup(
                oracle_state.checker,
                block_tokens,
                masks_suffix,
                str_prune,
                deadline,
                logits_pair,
                lp,
                tr,
            )
            smart_extra["smart_dedup_ms"] = round(
                (time.perf_counter() - t_d0) * 1000, 3
            )
            smart_extra["smart_dedup_true"] = bool(dedup_ok)
            if dedup_ok:
                result = True
                smart_extra["smart_dfs_ms"] = 0.0
            else:
                t_f0 = time.perf_counter()
                result = _oracle_dfs(
                    oracle_state.checker,
                    block_tokens,
                    0,
                    masks_suffix,
                    str_prune,
                    deadline,
                    logits_pair,
                    lp,
                    tr,
                )
                smart_extra["smart_dfs_ms"] = round(
                    (time.perf_counter() - t_f0) * 1000, 3
                )
        else:
            result = _oracle_dfs(
                oracle_state.checker,
                block_tokens,
                0,
                masks_suffix,
                str_prune,
                deadline,
                logits_pair,
                lp,
                tr,
            )
    except OracleSearchTimeout:
        search_ms = (time.perf_counter() - t_s0) * 1000
        actual_len = len(oracle_state.checker.tokens)
        if actual_len > prefix_len:
            oracle_state.checker.rollback(actual_len - prefix_len)
        oracle_state._synced_len = len(oracle_state.checker.tokens)
        _emit_timing(
            prefix_ms=prefix_ms,
            search_ms=search_ms,
            skipped=True,
            skip_reason="oracle_search_timeout",
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            max_search_seconds=max_s,
            logits_priority=lp,
            logits_priority_active=logits_active,
            trie_order=tr,
            trie_order_active=tr,
        )
        return None

    search_ms = (time.perf_counter() - t_s0) * 1000

    actual_len = len(oracle_state.checker.tokens)
    if actual_len != prefix_len:
        if actual_len > prefix_len:
            oracle_state.checker.rollback(actual_len - prefix_len)
        oracle_state._synced_len = len(oracle_state.checker.tokens)
        _emit_timing(
            prefix_ms=prefix_ms,
            search_ms=search_ms,
            checker_length_mismatch=True,
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=logits_active,
            trie_order=tr,
            trie_order_active=tr,
        )
        return None

    _emit_timing(
        prefix_ms=prefix_ms,
        search_ms=search_ms,
        n_masks=n_masks,
        string_prune_threshold=str_prune,
        logits_priority=lp,
        logits_priority_active=logits_active,
        trie_order=tr,
        trie_order_active=tr,
        **smart_extra,
    )
    return result


def oracle_find_block_assignment(
    oracle_state: OracleState,
    prefix_tokens: list[int],
    block_tokens: list[int],
    oracle_mask_limit: int,
    *,
    search_mode: Optional[SearchMode] = None,
    timing_out: Optional[Dict[str, Any]] = None,
    string_prune_threshold: Optional[int] = None,
    max_search_seconds: Optional[float] = None,
    model_logits: Optional[Any] = None,
    logits_index_base: Optional[int] = None,
    logits_priority: Optional[bool] = None,
    trie_order: Optional[bool] = None,
) -> Optional[list[int]]:
    """
    Like ``oracle_verify_fast`` but returns a **concrete block token list** (length
    ``len(block_tokens)``) when the block is satisfiable, else ``None``.

    * ``search_mode=dfs`` (default): single ``_oracle_dfs_assignment`` pass (sound).

    * ``search_mode=smart``: phase-1 ``bfs_dedup`` then sound ``_oracle_dfs`` boolean
      when dedup is False — if still unsat, return ``None`` **without** running
      assignment DFS (faster TN). If satisfiable, phase-2 ``_oracle_dfs_assignment``
      fills concrete tokens (logits ordering applies). When phase-1 dedup returns True,
      phase-2 assignment DFS still runs (dedup does not carry token choices).

    ``bfs`` / ``bfs_dedup`` alone are not supported here (``skip_reason=assignment_requires_dfs``).
    """
    t0 = time.perf_counter()
    mode = search_mode if search_mode is not None else _default_search_mode()
    lp = _resolve_logits_priority(logits_priority)
    tr = _resolve_trie_order(trie_order)
    str_prune = _string_prune_threshold_resolved(string_prune_threshold)
    if str_prune is not None:
        _warn_string_prune_once(str_prune)

    def _emit_timing(
        *,
        prefix_ms: float,
        search_ms: float,
        **extra: Any,
    ) -> None:
        if timing_out is None:
            return
        timing_out.clear()
        timing_out.update(
            {
                "prefix_sync_ms": round(prefix_ms, 3),
                "search_ms": round(search_ms, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000, 3),
                "search_mode": mode,
            }
        )
        timing_out.update(extra)

    n_masks = sum(1 for t in block_tokens if t == mask_id)
    if n_masks > oracle_mask_limit:
        _emit_timing(
            prefix_ms=0.0,
            search_ms=0.0,
            skipped=True,
            skip_reason="oracle_mask_limit",
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=False,
            trie_order=tr,
            trie_order_active=False,
            assignment_found=False,
        )
        return None

    if mode not in ("dfs", "smart"):
        _emit_timing(
            prefix_ms=0.0,
            search_ms=0.0,
            skipped=True,
            skip_reason="assignment_requires_dfs",
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=False,
            trie_order=tr,
            trie_order_active=False,
            assignment_found=False,
        )
        return None

    t_sync0 = time.perf_counter()
    sync_ok = oracle_state.sync_to_prefix(prefix_tokens)
    prefix_ms = (time.perf_counter() - t_sync0) * 1000
    if not sync_ok:
        _emit_timing(
            prefix_ms=prefix_ms,
            search_ms=0.0,
            sync_failed=True,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=False,
            trie_order=tr,
            trie_order_active=False,
            assignment_found=False,
        )
        return None

    prefix_len = len(oracle_state.checker.tokens)
    masks_suffix = _suffix_mask_counts(block_tokens)

    logits_pair: Optional[Tuple[Any, int]] = None
    if model_logits is not None and logits_index_base is not None:
        logits_pair = (model_logits, int(logits_index_base))
    logits_active = bool(lp and logits_pair is not None)

    max_s = _resolve_max_search_seconds(max_search_seconds)
    deadline = (time.perf_counter() + max_s) if max_s is not None else None

    t_s0 = time.perf_counter()
    smart_extra: Dict[str, Any] = {}
    try:
        if mode == "smart":
            _warn_smart_once()
            _warn_bfs_dedup_once()
            t_d0 = time.perf_counter()
            dedup_ok = _oracle_bfs_dedup(
                oracle_state.checker,
                block_tokens,
                masks_suffix,
                str_prune,
                deadline,
                logits_pair,
                lp,
                tr,
            )
            smart_extra["smart_dedup_ms"] = round(
                (time.perf_counter() - t_d0) * 1000, 3
            )
            smart_extra["smart_dedup_true"] = bool(dedup_ok)
            if dedup_ok:
                smart_extra["smart_dfs_ms"] = 0.0
            else:
                t_f0 = time.perf_counter()
                dfs_sat = _oracle_dfs(
                    oracle_state.checker,
                    block_tokens,
                    0,
                    masks_suffix,
                    str_prune,
                    deadline,
                    logits_pair,
                    lp,
                    tr,
                )
                smart_extra["smart_dfs_ms"] = round(
                    (time.perf_counter() - t_f0) * 1000, 3
                )
                if not dfs_sat:
                    search_ms = (time.perf_counter() - t_s0) * 1000
                    actual_len = len(oracle_state.checker.tokens)
                    if actual_len != prefix_len:
                        if actual_len > prefix_len:
                            oracle_state.checker.rollback(actual_len - prefix_len)
                        oracle_state._synced_len = len(oracle_state.checker.tokens)
                    _emit_timing(
                        prefix_ms=prefix_ms,
                        search_ms=search_ms,
                        n_masks=n_masks,
                        string_prune_threshold=str_prune,
                        logits_priority=lp,
                        logits_priority_active=logits_active,
                        trie_order=tr,
                        trie_order_active=tr,
                        assignment_found=False,
                        **smart_extra,
                    )
                    return None

            t_a0 = time.perf_counter()
            full = _oracle_dfs_assignment(
                oracle_state.checker,
                block_tokens,
                0,
                masks_suffix,
                str_prune,
                deadline,
                logits_pair,
                lp,
                tr,
            )
            smart_extra["smart_assignment_ms"] = round(
                (time.perf_counter() - t_a0) * 1000, 3
            )
        else:
            full = _oracle_dfs_assignment(
                oracle_state.checker,
                block_tokens,
                0,
                masks_suffix,
                str_prune,
                deadline,
                logits_pair,
                lp,
                tr,
            )
    except OracleSearchTimeout:
        search_ms = (time.perf_counter() - t_s0) * 1000
        actual_len = len(oracle_state.checker.tokens)
        if actual_len > prefix_len:
            oracle_state.checker.rollback(actual_len - prefix_len)
        oracle_state._synced_len = len(oracle_state.checker.tokens)
        _emit_timing(
            prefix_ms=prefix_ms,
            search_ms=search_ms,
            skipped=True,
            skip_reason="oracle_search_timeout",
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            max_search_seconds=max_s,
            logits_priority=lp,
            logits_priority_active=logits_active,
            trie_order=tr,
            trie_order_active=tr,
            assignment_found=False,
            **smart_extra,
        )
        return None

    search_ms = (time.perf_counter() - t_s0) * 1000

    actual_len = len(oracle_state.checker.tokens)
    if actual_len != prefix_len:
        if actual_len > prefix_len:
            oracle_state.checker.rollback(actual_len - prefix_len)
        oracle_state._synced_len = len(oracle_state.checker.tokens)
        _emit_timing(
            prefix_ms=prefix_ms,
            search_ms=search_ms,
            checker_length_mismatch=True,
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=logits_active,
            trie_order=tr,
            trie_order_active=tr,
            assignment_found=False,
            **smart_extra,
        )
        return None

    if full is None or len(full) != len(block_tokens):
        _emit_timing(
            prefix_ms=prefix_ms,
            search_ms=search_ms,
            n_masks=n_masks,
            string_prune_threshold=str_prune,
            logits_priority=lp,
            logits_priority_active=logits_active,
            trie_order=tr,
            trie_order_active=tr,
            assignment_found=False,
            **smart_extra,
        )
        return None

    _emit_timing(
        prefix_ms=prefix_ms,
        search_ms=search_ms,
        n_masks=n_masks,
        string_prune_threshold=str_prune,
        logits_priority=lp,
        logits_priority_active=logits_active,
        trie_order=tr,
        trie_order_active=tr,
        assignment_found=True,
        **smart_extra,
    )
    return full


# Module-level state set by ``init_oracle`` (per instance, before ``lave_generate``).
_ORACLE_STATE: Optional[OracleState] = None
_ORACLE_MASK_LIMIT: int = 12


def init_oracle(grammar_str: str, oracle_mask_limit: int = 12) -> None:
    global _ORACLE_STATE, _ORACLE_MASK_LIMIT
    _ORACLE_STATE = OracleState(grammar_str)
    _ORACLE_MASK_LIMIT = oracle_mask_limit


def get_oracle_state() -> Optional[OracleState]:
    return _ORACLE_STATE


def get_oracle_mask_limit() -> int:
    return _ORACLE_MASK_LIMIT

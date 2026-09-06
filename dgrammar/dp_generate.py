"""DP-based grammar-constrained generation for diffusion LMs.

Instead of greedily retrying one violation at a time (generate.py), this module
runs a Viterbi DP over grammar DFA states across all non-mask positions in the
sequence. Key properties:

  - State = grammar DFA node, identified by bytes(matcher.compute_logit_bias()).
    Two token paths that land in the same DFA state are merged (Viterbi-style),
    keeping only the higher log-prob path. This bounds active states by DFA size,
    not by vocab^k.

  - Per position: O(|states| * top_k) rollback/advance probes (no cloning),
    then O(|next_states|) deep_copy() calls (one clone per surviving DFA state).

  - Returns the globally optimal token assignment for the entire non-mask
    prefix segment, rather than fixing one violator at a time.

Typical DFA size for JSON-schema grammars: 20–200 states, so the DP is fast.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from dgrammar.checker import TokenChecker
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens


# ── Core DP ──────────────────────────────────────────────────────────────────


def only_stop_remains(matcher, eos_id: int, eot_id: int) -> bool:
    """True when the parser has no consumable transition left.

    EOS is not a grammar symbol.  ``compute_logit_bias()`` marks it allowed at
    an accepting state -- "you may stop here" -- while ``try_consume_tokens``
    refuses it, because there is nothing to advance over.  So a finished
    document arrives at the frontier looking like a violation, and something
    has to recognise it; without that the repair cascade tries to fix the EOS,
    fails, remasks, the sampler puts EOS back, and the tail grinds (measured:
    dead ends 33 -> 246 and validity -5.3pp when the recogniser was removed).

    ``is_accepting()`` is the wrong test for this.  It says the parser *may*
    stop, not that it *must*: on jsb_medium the two agree (53 of 53 accepting
    states allowed EOS and nothing else), but a SMILES accepting state allows
    1388 tokens, so stopping there takes the length decision away from the
    model -- every molecule came out as `C` or `CC`.  Asking whether anything
    consumable is left is the same question for both grammars.
    """
    arr = np.frombuffer(matcher.compute_logit_bias(), dtype=np.uint8)
    n_legal = int(np.count_nonzero(arr))
    if n_legal == 0:
        # A live state always has a continuation -- 2028 of 2028 sampled states
        # had a non-empty mask.  An empty one means the read is wrong, not that
        # the document is over.  Treating it as "done" ended one instance at
        # step 77 of 128 with 103 positions still masked and the parser one
        # token short of a valid document.
        return False
    if n_legal > 2:
        return False         # more than EOS+EOT are legal, so it can continue
    if not all(t in (eos_id, eot_id) for t in np.nonzero(arr)[0].tolist()):
        return False
    # Costs nothing measurable (0.000 ms) and makes the claim unfalsifiable by
    # a bad mask read: a document that is not syntactically complete can never
    # be declared finished, whatever the bias says.
    return matcher.is_accepting()


def find_constraint_end(
    matcher,
    x: torch.Tensor,
    start_pos: int,
    mask_id: int,
    max_lookahead: int = 48,
    open_tok_ids: Optional[set] = None,
    close_tok_ids: Optional[set] = None,
    init_depth: int = 0,
    out: Optional[dict] = None,
) -> int:
    """Find the end of the violated constraint starting at start_pos.

    Probes forward using the grammar automaton with a deep copy of the matcher.
    Bracket depth is tracked using the ORIGINAL tokens in x (not probe substitute
    tokens), starting from init_depth (computed by the caller by scanning the
    already-consumed prefix from gen_start to start_pos).

    Junction condition: original token accepted by grammar AND bracket depth == 0
    after processing that token.  This prevents the DP span from stopping inside
    an unclosed array or object (which would allow the DP to collapse the
    remainder to [] or {}).

    Returns the exclusive end position c_end such that DP runs on
    [start_pos, c_end) and original tokens [c_end, ...) are kept.
    Falls back to start_pos + max_lookahead if no junction found.

    If ``out`` is given it is filled with ``reason`` -- why the scan stopped --
    which distinguishes the case the junction condition was actually met
    ("junction") from the three exits that return the raw cap without ever
    testing depth ("mask", "dead", "lookahead").  Measurement only; the return
    value is unchanged.
    """
    seq_len = x.shape[1]
    probe = matcher.deep_copy()
    end = min(seq_len, start_pos + max_lookahead)
    depth = init_depth   # net unclosed brackets (accurate from caller's backward scan)

    for pos in range(start_pos, end):
        tid = x[0, pos].item()
        if tid == mask_id:
            if out is not None:
                out["reason"] = "mask"
                out["stop_pos"] = pos
                out["stop_depth"] = depth
            break

        # Update depth from the ORIGINAL token first, before checking junction.
        # This prevents stopping at `[` or `{` (depth just went up), and allows
        # stopping at `]` / `}` that closes back to depth 0.
        if open_tok_ids and tid in open_tok_ids:
            depth += 1
        elif close_tok_ids and tid in close_tok_ids:
            depth = max(0, depth - 1)

        consumed = probe.try_consume_tokens([tid])
        if consumed == 1:
            # Original token accepted by grammar.
            if depth == 0:
                # Depth just returned to 0 (or was never in a bracket) — safe junction.
                if out is not None:
                    out["reason"] = "junction"
                    out["stop_pos"] = pos
                    out["stop_depth"] = 0
                return pos
            # Still inside an open bracket — keep scanning.
        else:
            probe.rollback(0)
            # Original token rejected — advance probe with any valid token to keep
            # the automaton state moving.  We do NOT update depth here because the
            # substitute token is artificial; depth tracking uses only original tokens.
            advanced = False
            for candidate in range(256):
                c2 = probe.try_consume_tokens([candidate])
                if c2 == 1:
                    advanced = True
                    break
                probe.rollback(0)
            if not advanced:
                if out is not None:
                    out["reason"] = "dead"
                    out["stop_pos"] = pos
                    out["stop_depth"] = depth
                break

    if out is not None and "reason" not in out:
        out["reason"] = "lookahead"
        out["stop_pos"] = end
        out["stop_depth"] = depth
    return min(end, seq_len)


def dp_fix_prefix(
    matcher,
    x: torch.Tensor,
    start_pos: int,
    log_probs: torch.Tensor,
    mask_id: int,
    top_k: int = 50,
    max_positions: int = 64,
    end_pos: Optional[int] = None,
    include_masked: bool = False,
    eos_id: int = -1,
    eot_id: int = -1,
    objective: str = "logp",
    cand_source: str = "automaton",
    eos_in_candidates: bool = False,
    beam_per_key: Optional[int] = 1,
    max_live: int = 4096,
    out: Optional[dict] = None,
) -> tuple[Optional[list[tuple[int, int]]], int]:
    """Find a grammar-valid token assignment for the span starting at start_pos.

    Maximises the sum of log p_theta over the span, subject to the grammar
    accepting the whole span.

    Two objectives:

    ``logp``
        maximise the sum of log p_theta over the span.

    ``min_edit``
        lexicographic: fewest positions changed from what the model wrote,
        ties broken by log-probability.

    ``logp`` asks "what is the most probable valid string over this span?" when
    the job is to *repair one violation*, so it rewrites positions the grammar
    never objected to: on jsb_medium it changed 5.95 of a 14-position span
    while only the violator was illegal.  One traced instance had the repair
    replace a run of correct values with other schema key names, turning a
    parseable document into `"orderid": "price"`.

    This objective was removed once already, measured at 0.4pp against
    ``logp``.  That measurement was taken when the repair window was one token
    wide -- with a single position to place, "fewest changes" and "most
    probable" pick the same token -- and when half of all DP calls dead-ended
    before making any edit at all.  Neither is true now: the span is the full
    constraint region and dead ends are gone, so the edit count has room to
    differ.  The scalar ``deviation_penalty`` this generalises stayed removed;
    it needed a tuned constant, and ordering the two terms needs none.

    Candidates at each position are drawn from the automaton -- the tokens the
    parser can actually consume from the current state, ranked by the model --
    rather than from the model's top-k over the whole vocabulary.  ``top_k``
    therefore bounds how many legal alternatives are weighed, not whether a
    legal one is found at all: where the state allows fewer than ``top_k``
    tokens the edge set is enumerated exactly.

    The token already at each position is also offered whenever the grammar
    allows it, so "leave this position as the model wrote it" stays reachable
    even when ``top_k`` does not rank it.

    When include_masked=True, also processes MASK positions, so a span may be
    filled rather than only rewritten.  The oracle probe uses this.

    Args:
        matcher: LLMatcher at the grammar state just before start_pos.
                 Pass ``checker.matcher.deep_copy()`` — this matcher is mutated.
        x: Current token sequence, shape [1, seq_len].
        start_pos: First position to process (inclusive). Stop at first mask.
        log_probs: Log-softmax probabilities, shape [1, seq_len, vocab_size].
        mask_id: Token ID for the MASK token.
        top_k: How many of the state's legal tokens to explore at each position,
            taken in order of model probability.  A smaller value than the
            vocabulary-wide version needed: the candidates are all legal, so
            none of the budget is spent on tokens the parser would reject.
        eos_id, eot_id: Stop tokens.  Excluded from the candidate set because
            the parser cannot consume them -- offering them wastes a slot and,
            where they are the only legal token, empties the candidate set and
            looks like a dead end.  Reaching such a state means the span ends
            there, which is what the second return value reports.
        out: Filled with ``n_positions`` and, on a dead end, ``dead_step`` -- the
            layer at which no candidate was accepted from any reachable state.
            Everything before that layer already has a valid assignment sitting
            in ``states``; this records how much of it the all-or-nothing
            contract is throwing away.  Measurement only; the return value is
            unchanged.
        beam_per_key: how many distinct paths to keep per state key.  ``1`` is
            the published behaviour: paths that leave the parser in states with
            the same legal-token mask collapse to the best one.  The key is an
            abstraction, not an identifier -- two states can admit the same
            tokens and still differ later -- so collapsing to one is sound but
            not exact.  Larger values keep more of them; ``None`` disables
            merging entirely, which searches the candidate lattice exhaustively
            and therefore returns its true optimum.  Only the search changes;
            every path is still one the parser consumed, at any setting.
        max_live: global cap on paths carried into the next layer.  Only binds
            when ``beam_per_key`` is large or ``None``; ``out["capped"]``
            records whether it ever did, which is what tells an exhaustive run
            from a truncated one.
        objective: "logp" or "min_edit". See above.
        end_pos: Exclusive upper bound for the DP span. When provided (from
            find_constraint_end), the DP operates only on [start_pos, end_pos)
            and the caller resumes greedy extension from end_pos onwards.
            If None, the span extends to the next mask token (original behaviour).

    Returns:
        ``(replacements, reached_end)``.  ``replacements`` lists
        (position, new_token_id) for positions whose optimal token differs from
        the current x[0, pos], or [] if nothing needs changing, or None if not
        even the first position could be assigned.  ``reached_end`` is the
        exclusive position the assignment actually covers: normally ``end_pos``,
        and earlier when the parser reached a state with nothing left to
        consume, in which case the document ends there.
    """
    NEG_INF = -math.inf

    # Collect positions starting at start_pos.
    # In normal mode: stop at the first MASK token (repair mode).
    # In include_masked mode: include MASK tokens too.
    seq_len = x.shape[1]
    hard_end = end_pos if end_pos is not None else seq_len
    positions: list[int] = []
    p = start_pos
    while p < hard_end and p < seq_len:
        if x[0, p].item() == mask_id and not include_masked:
            break
        positions.append(p)
        p += 1

    if not positions:
        if out is not None:
            out["n_positions"] = 0
        return [], start_pos

    # Cap segment length to avoid O(|states|×top_k×seg_len) blowup on long segments.
    if len(positions) > max_positions:
        positions = positions[:max_positions]
    if out is not None:
        out["n_positions"] = len(positions)

    # ── DP initialisation ────────────────────────────────────────────────────
    # states: state_key → (matcher_clone, cumulative_score)
    # back:   (step_index, state_key) → (prev_state_key, chosen_token_id)
    #
    # state_key = bytes(compute_logit_bias()) is a proxy for the DFA node.
    # Two paths reaching the same DFA node are merged; only the best survives.

    init_key: bytes = bytes(matcher.compute_logit_bias())
    # states is a list of live paths, each (matcher, edits, score, key).
    # `edits` counts positions whose token differs from what the model wrote.
    # With beam_per_key=1 there is exactly one path per key and the list is the
    # old dict in insertion order, so the search is bit-identical to it.
    states: list[tuple] = [(matcher, 0, 0.0, init_key)]
    back: dict[tuple, tuple] = {}       # (step, path_index) -> (prev_index, tok)


    def better(cand, held) -> bool:
        """Is candidate (edits, score) preferable to the held one?"""
        if objective == "min_edit":
            return (cand[0], -cand[1]) < (held[0], -held[1])
        return cand[1] > held[1]

    def offer(bucket: list, entry: tuple) -> None:
        """Add (prev_idx, tok, edits, score) to a key's bucket under the beam.

        At beam_per_key=1 this is exactly the old "replace the held winner iff
        strictly better", so ties still go to the path seen first.
        """
        if beam_per_key is None or len(bucket) < beam_per_key:
            bucket.append(entry)
            return
        worst = 0
        for i in range(1, len(bucket)):
            if better(bucket[worst][2:4], bucket[i][2:4]):
                worst = i
        if better(entry[2:4], bucket[worst][2:4]):
            bucket[worst] = entry

    capped = False
    n_live_max = 1

    # ── DP loop ──────────────────────────────────────────────────────────────
    n_done = len(positions)
    for step, pos in enumerate(positions):
        pos_lp = log_probs[0, pos]          # [vocab_size]
        pos_lp_np = pos_lp.detach().to("cpu").numpy()   # one transfer per position
        vocab = pos_lp_np.shape[0]
        k = min(top_k, vocab)
        orig_tok = x[0, pos].item()

        # Phase 1 — exploration via rollback (no cloning).
        # For each active (prev_state, candidate_token) pair:
        #   - try consuming the token
        #   - record new DFA state and score if it beats the current winner
        #   - rollback to restore prev_state
        winners: dict[bytes, list] = {}   # new_key -> [(prev_idx, tok, edits, score)]

        for prev_idx, (prev_m, prev_edits, prev_score, _prev_key) in enumerate(states):
            # Candidates come from the automaton, not from the model's ranking.
            #
            # compute_logit_bias() is this state's outgoing edge set -- one byte
            # per token, nonzero meaning the parser can consume it.  Ranking the
            # whole vocabulary by probability and then asking the parser which of
            # the top k it accepts is rejection sampling against that set, and it
            # fails whenever the set is small: on jsb_medium 46% of live states
            # allow fewer than 100 of 126,349 tokens, so a top-100 draw can miss
            # them entirely and the DP reports "no path" where a path exists.
            # Drawing from the edge set instead makes that impossible -- a live
            # state always has at least one legal token (0 of 2028 sampled states
            # had an empty set) -- and where the set is smaller than k it is
            # enumerated exactly rather than sampled.
            arr = np.frombuffer(prev_m.compute_logit_bias(), dtype=np.uint8)
            if arr.shape[0] > vocab:
                arr = arr[:vocab]
            if cand_source == "vocab":
                # Ablation only: rank the whole vocabulary and keep whichever of
                # the top k the parser accepts.  This is the filter-then-propose
                # order the paper argues against, kept behind a flag so the two
                # candidate sources can be measured with one binary instead of
                # two code versions.
                top = np.argpartition(-pos_lp_np, k - 1)[:k]
                legal_ids = top[arr[top] != 0]
            else:
                legal_ids = np.nonzero(arr)[0]
            # EOS/EOT are marked legal at an accepting state but cannot be
            # consumed; keeping them only burns candidate slots.
            if legal_ids.size and not eos_in_candidates:
                legal_ids = legal_ids[(legal_ids != eos_id) & (legal_ids != eot_id)]
            if out is not None:
                out["legal_total"] = out.get("legal_total", 0) + int(legal_ids.size)
                out["legal_calls"] = out.get("legal_calls", 0) + 1
                if legal_ids.size <= k:
                    out["legal_exact"] = out.get("legal_exact", 0) + 1
                out["legal_min"] = min(out.get("legal_min", 1 << 30),
                                       int(legal_ids.size))
            if legal_ids.size == 0:
                continue
            if legal_ids.size > k:
                sel = np.argpartition(-pos_lp_np[legal_ids], k - 1)[:k]
                cand_ids = legal_ids[sel]
            else:
                cand_ids = legal_ids
            # Keep "leave this position as the model wrote it" reachable when the
            # grammar allows it, even if k does not rank it.
            if (orig_tok != mask_id and orig_tok < arr.shape[0] and arr[orig_tok]
                    and not (cand_ids == orig_tok).any()):
                cand_ids = np.append(cand_ids, orig_tok)

            for tid in cand_ids.tolist():
                lp = float(pos_lp_np[tid])
                if lp == NEG_INF:
                    continue

                consumed = prev_m.try_consume_tokens([tid])
                if consumed < 1:
                    # The bias said this token was legal but the matcher refused
                    # it.  Should never happen; counted so it cannot hide.
                    if out is not None:
                        out["bias_disagreements"] = out.get("bias_disagreements", 0) + 1
                    prev_m.rollback(0)   # safe no-op via checker.rollback guard
                    continue

                is_orig = (orig_tok != mask_id and tid == orig_tok)
                new_score = prev_score + lp
                new_edits = prev_edits + (0 if is_orig else 1)
                new_key = bytes(prev_m.compute_logit_bias())
                prev_m.rollback(1)   # restore prev_m for the next candidate

                offer(winners.setdefault(new_key, []),
                      (prev_idx, tid, new_edits, new_score))

        if not winners:
            # Nothing consumable at this position from any reachable state.
            # With the stop tokens excluded that means the parser is finished
            # here, so the span ends rather than fails: positions[0..step-1]
            # already have a valid assignment and are returned.
            if out is not None:
                out["dead_step"] = step
            n_done = step
            break

        # A global cap so that beam_per_key=None cannot blow up on a wide span.
        # Whether it ever bound is what separates an exhaustive run from a
        # truncated one, so it is recorded rather than silently applied.
        flat = [(key, e) for key, bucket in winners.items() for e in bucket]
        if len(flat) > max_live:
            capped = True
            flat.sort(key=(lambda ke: (ke[1][2], -ke[1][3]))
                      if objective == "min_edit" else (lambda ke: -ke[1][3]))
            flat = flat[:max_live]
        n_live_max = max(n_live_max, len(flat))

        # Phase 2 — clone one matcher per surviving path.
        next_states: list[tuple] = []
        for new_key, (prev_idx, tok_id, new_edits, new_score) in flat:
            prev_m = states[prev_idx][0]
            new_m = prev_m.deep_copy()
            consumed = new_m.try_consume_tokens([tok_id])
            assert consumed == 1, (
                f"Phase-2 replay failed for token {tok_id} at pos {pos}: "
                f"expected 1 consumed, got {consumed}"
            )
            back[(step, len(next_states))] = (prev_idx, tok_id)
            next_states.append((new_m, new_edits, new_score, new_key))

        states = next_states

    # ── Backtrack ─────────────────────────────────────────────────────────────
    if not states or n_done == 0:
        return None, start_pos

    if objective == "min_edit":
        best_idx = min(range(len(states)), key=lambda i: (states[i][1], -states[i][2]))
    else:
        best_idx = max(range(len(states)), key=lambda i: states[i][2])

    if out is not None:
        out["best_score"] = states[best_idx][2]
        out["best_edits"] = states[best_idx][1]
        out["capped"] = capped
        out["n_live_max"] = n_live_max
        # Exhaustive over the candidate lattice: nothing was merged away and
        # nothing was dropped by the global cap, so this IS the lattice optimum.
        out["exhaustive"] = (beam_per_key is None) and not capped

    replacements: list[tuple[int, int]] = []
    assignment: list[int] = []
    cur = best_idx
    for step in range(n_done - 1, -1, -1):
        prev_idx, tok_id = back[(step, cur)]
        orig_tok = x[0, positions[step]].item()
        assignment.append(tok_id)
        if tok_id != orig_tok:
            replacements.append((positions[step], tok_id))
        cur = prev_idx

    replacements.reverse()
    if out is not None:
        assignment.reverse()
        out["assignment"] = assignment
    return replacements, positions[n_done - 1] + 1


# ── Oracle probe: how much does freezing the prefix cost? ────────────────────


#: W = 0 pinned to what repair actually searched. Feasibility floor, not a
#: sweep arm -- it is the only arm whose search space provably contains the
#: answer repair settled for.
REPAIR_ARM = "0@rw"


class SpanStats:
    """Where does the repair span actually end, and at what bracket depth?

    ``find_constraint_end`` documents a junction condition -- grammar accepts
    AND bracket depth back to 0 -- whose stated purpose is to stop the DP span
    from ending inside an unclosed array or object, "which would allow the DP
    to collapse the remainder to [] or {}".  But three of its four exits
    (``mask``, ``dead``, ``lookahead``) return the raw ``start_pos +
    max_lookahead`` cap without ever testing depth, and repair's
    ``dp_fix_prefix`` runs with ``include_masked=False`` so it truncates again
    at the first MASK.  The span the DP actually optimises over can therefore
    end anywhere, including at depth > 0.

    This records, per repair site, where the span really ended and what the
    bracket depth was there -- separating "the guard held" from "the guard was
    never consulted".  Measurement only: no forward pass, no extra DP, no
    change to token placement, so it is cheap enough to leave on for every arm.

    ``depth_before`` is computed from the tokens the model placed;
    ``depth_after`` from the tokens the DP chose.  ``depth_after`` <
    ``depth_before`` means the repair net-closed brackets that the model had
    left open -- the collapse, caught in the act.
    """

    def __init__(self) -> None:
        self.sites: list[dict] = []
        #: Why a position was handed back to the sampler.  The four paths cost
        #: the same (one remask, one more forward pass) but mean different
        #: things, and only the totals were ever recorded -- 75 DP calls and 32
        #: dead-ends against 139 resamples left two thirds unaccounted for.
        #:
        #:   single_token      a one-token substitution parsed, then the greedy
        #:                     resume hit a further violation downstream
        #:   dp_span_replay    DP returned a fix, but replaying the span through
        #:                     the real matcher consumed fewer tokens than the DP
        #:                     believed it would
        #:   dp_suffix_replay  the span replayed, but the original tokens after
        #:                     it no longer parse -- the DP chose an end state
        #:                     the rest of the output cannot continue from
        #:   dp_dead_end       DP found no grammar-valid path at all
        self.resample_reasons: dict[str, int] = {}
        self.violations: list[dict] = []
        #: Which branch ended generation.  o21142 stopped at step 77 of 128 with
        #: 102 positions still masked, one violation and no handback, and none
        #: of the branches then believed to be responsible could have fired --
        #: removing the one blamed for it changed nothing on any of 64
        #: instances.  Recording the branch instead of inferring it.
        self.stop: dict = {}

    def record(self, **kw) -> None:
        self.sites.append(kw)

    def resample(self, reason: str) -> None:
        self.resample_reasons[reason] = self.resample_reasons.get(reason, 0) + 1

    def stopped(self, reason: str, **kw) -> None:
        if not self.stop:            # first one wins; later steps cannot re-end it
            self.stop = dict(reason=reason, **kw)

    def violation(self, **kw) -> None:
        """What the model wrote at a violator, against what the grammar allows.

        87% of repair sites reach a position where the grammar permits at most
        one token, and there the DP is not choosing -- it transcribes the
        grammar's demand over whatever the model wrote.  One such case was
        traced by hand: llguidance orders object properties by their order of
        declaration in the schema, so a model that writes `"orderid"` where the
        schema declares `"item"` next produces a violation that is not an error
        at all -- both key orders satisfy the schema, and jsonschema accepts
        either.  Repairing it in place is impossible (the fix is to move a
        block, which the DP cannot express), so the repair mislabels the field.

        This records the pair so the share of violations that are ordering
        artifacts can be counted instead of extrapolated from one instance.
        """
        if len(self.violations) < 200:
            self.violations.append(kw)

    def summary(self) -> dict:
        n = len(self.sites)
        if n == 0:
            return {"n_sites": 0}

        def frac(pred) -> float:
            return 100.0 * sum(1 for r in self.sites if pred(r)) / n

        out: dict = {"n_sites": n}
        for reason in ("junction", "mask", "dead", "lookahead"):
            out[f"reason_{reason}_pct"] = frac(lambda r, _x=reason: r["reason"] == _x)

        # The population the guard was supposed to protect and did not reach.
        out["unsafe_end_pct"] = frac(
            lambda r: r["reason"] != "junction" and r["depth_before"] > 0
        )
        out["net_closed_pct"] = frac(lambda r: r["depth_after"] < r["depth_before"])
        out["eos_inserted_pct"] = frac(lambda r: r["eos_in_fixes"])
        out["dp_failed_pct"] = frac(lambda r: r["n_fixes"] is None)

        spans = [r["span"] for r in self.sites]
        out["mean_span"] = sum(spans) / n
        out["span_eq1_pct"] = frac(lambda r: r["span"] == 1)
        out["mean_depth_before"] = sum(r["depth_before"] for r in self.sites) / n
        out["mean_depth_drop"] = sum(
            r["depth_before"] - r["depth_after"] for r in self.sites
        ) / n
        # How much of constraint_end the DP never got to see, because
        # dp_fix_prefix truncated at the first MASK.
        out["mean_truncated_by_mask"] = sum(
            r["constraint_end"] - r["eff_end"] for r in self.sites
        ) / n
        out["resample_reasons"] = dict(self.resample_reasons)
        out["n_violations"] = len(self.violations)
        return out


class OracleStats:
    """How far back must the prefix be unfrozen before a better answer appears?

    Repair searches ``[violator, end)`` with everything earlier already consumed
    by the parser and therefore immutable.  The oracle re-runs the *same* DP --
    same grammar, same probabilities, same span end -- but unfreezes the last
    ``W`` positions before the violator.

    ``W = 0`` unfreezes nothing and is the control; ``W = None`` frees the whole
    generation.  Sweeping W turns a yes/no question ("was the optimum reachable")
    into a curve ("how far back must one go"), and the control is what makes the
    curve mean anything: every arm runs the same code with the same ``top_k`` and
    the same span end, so the only variable is how much prefix is released.

    Reading it against ``W = 0`` matters because the naive full-window oracle
    also enlarges the search budget -- on a pilot it searched up to 242 positions
    against a repair that had settled for one -- and improvement from a bigger
    search is not evidence that freezing excluded anything.

    A repair site counts as *excluded* at W when the oracle prefers to change a
    position before the violator: the better answer is outside the set repair can
    search, so no repair can recover it.

    Alongside the sweep runs ``REPAIR_ARM``: W = 0 pinned to ``repair_end``
    rather than ``span_end``.  That arm is not part of the sweep -- it varies
    the forward reach as well -- it is the feasibility floor.  It searches
    exactly what repair searched, so it must succeed wherever repair did, and a
    sweep arm failing while it succeeds is the probe starving, not the grammar
    refusing.  The first pilot had no such floor: its W = 0 control solved 4 of
    24 sites while repair had succeeded at all 24, and nothing in the output
    distinguished "freezing excluded the answer" from "the probe could not find
    the answer it was holding".

    The probe runs with repair's own ``top_k`` and
    penalty.  Anything else breaks nesting: repair's answer must lie inside the
    oracle's search space, or a wider window can return a *worse* string and
    ``gap_nats`` goes negative -- which is a bug in the measurement, never a
    property of the site.
    """

    def __init__(self, windows=(0, 8, 32, None)):
        # W = 0    -> nothing unfrozen, but searching forward to span_end. The
        #             single-variable control: same code path, same top_k, same
        #             span end as every other W, so the only thing that varies
        #             across the sweep is how far back the prefix is released.
        # W = None -> unfreeze everything back to gen_start.
        self.windows = tuple(windows)
        self.sites: list[dict] = []
        self.probe_time = 0.0
        self.failures = 0
        # Per-arm bookkeeping. A window key can be absent from a site for three
        # different reasons -- the DP found no path, the parser replay failed,
        # or the arm was never attempted -- and the first pilot could not tell
        # them apart, which is how a control that solved 4/24 sites went unread.
        self.attempts: dict[str, int] = {}
        self.arm_failures: dict[str, int] = {}
        self.replay_skips: dict[str, int] = {}

    def arm_keys(self) -> list[str]:
        return [REPAIR_ARM] + ["full" if w is None else str(w) for w in self.windows]

    def summary(self) -> dict:
        n = len(self.sites)
        out = {"oracle_sites": n, "oracle_failures": self.failures,
               "oracle_probe_s": self.probe_time,
               "oracle_windows": list(self.windows),
               "oracle_arms": self.arm_keys()}
        if n == 0:
            return out
        for key in self.arm_keys():
            rows = [s["by_window"][key] for s in self.sites if key in s["by_window"]]
            tried = self.attempts.get(key, 0)
            if tried:
                out[f"solved_pct_w{key}"] = 100.0 * len(rows) / tried
                out[f"nopath_w{key}"] = self.arm_failures.get(key, 0)
                out[f"replay_skip_w{key}"] = self.replay_skips.get(key, 0)
            if not rows:
                continue
            gaps = sorted(r["gap_nats"] for r in rows)
            out[f"excluded_pct_w{key}"] = 100.0 * sum(
                1 for r in rows if r["prefix_edits"] > 0) / len(rows)
            out[f"median_gap_nats_w{key}"] = gaps[len(gaps) // 2]
            out[f"mean_prefix_edits_w{key}"] = sum(
                r["prefix_edits"] for r in rows) / len(rows)
            # Nesting invariant. REPAIR_ARM and W=full both search supersets of
            # what repair searched, under the same top_k, so
            # neither can return a lower-scoring string. A nonzero count here
            # means the arms came apart again -- read nothing else until it is 0.
            out[f"neg_gap_w{key}"] = sum(1 for r in rows if r["gap_nats"] < -1e-9)
            out[f"mask_fill_nats_w{key}"] = sum(
                r.get("mask_fill_nats", 0.0) for r in rows) / len(rows)
            out[f"mean_filled_masks_w{key}"] = sum(
                r.get("filled_masks", 0) for r in rows) / len(rows)

        # Only the excess over a W=0 control is attributable to unfreezing; the
        # rest is what re-searching the same span buys on its own. Pair against
        # both controls: "0" holds span_end fixed and is the clean single-
        # variable comparison, REPAIR_ARM is the one guaranteed to be feasible.
        for ctrl in ("0", REPAIR_ARM):
            for key in self.arm_keys():
                if key == ctrl:
                    continue
                paired = [(s["by_window"][key]["gap_nats"], s["by_window"][ctrl]["gap_nats"])
                          for s in self.sites
                          if key in s["by_window"] and ctrl in s["by_window"]]
                if paired:
                    d = sorted(a - b for a, b in paired)
                    sfx = "" if ctrl == "0" else "_vs_repair"
                    out[f"unfreeze_gap_nats_w{key}{sfx}"] = d[len(d) // 2]
                    out[f"unfreeze_n_w{key}{sfx}"] = len(paired)
        return out


def _oracle_probe(
    initial_matcher,
    checker_matcher,
    x: torch.Tensor,
    gen_start: int,
    span_end: int,
    repair_end: int,
    log_probs: torch.Tensor,
    mask_id: int,
    violator: int,
    top_k: int,
    stats: OracleStats,
    eos_id: int = -1,
    eot_id: int = -1,
    objective: str = "logp",
    cand_source: str = "automaton",
    eos_in_candidates: bool = False,
) -> None:
    """Measurement only -- never mutates x, the checker, or the schedule.

    ``span_end`` is how far forward every arm searches (the full contiguous
    committed region); ``repair_end`` is where the progressive window actually
    stopped. Recording both separates the cost of the ``ws=1`` break-on-first-
    success heuristic from the cost of freezing the prefix.
    """
    t0 = time.perf_counter()
    try:
        if span_end <= violator:
            return
        record: dict = {"violator": violator - gen_start,
                        "span_end": span_end - gen_start,
                        "repair_width": repair_end - violator,
                        "by_window": {}}

        # (key, unfrozen width, forward end). The sweep holds the forward end at
        # span_end so W is the only thing that moves; REPAIR_ARM instead pins it
        # to repair_end, reproducing the search repair actually ran.
        arms = [(REPAIR_ARM, 0, repair_end)]
        arms += [("full" if w is None else str(w), w, span_end)
                 for w in stats.windows]

        for key, w, end in arms:
            if end <= violator:
                continue
            start = gen_start if w is None else max(gen_start, violator - w)
            stats.attempts[key] = stats.attempts.get(key, 0) + 1

            # Replay the parser to `start`. For the full window that is the
            # saved gen_start state; otherwise walk forward from it.
            m = initial_matcher.deep_copy()
            if start > gen_start:
                pre = [x[0, p].item() for p in range(gen_start, start)
                       if x[0, p].item() != mask_id]
                if pre and m.try_consume_tokens(pre) != len(pre):
                    # Prefix no longer replays. Not a no-path result -- record it
                    # apart so a starved arm is never read as an excluded one.
                    stats.replay_skips[key] = stats.replay_skips.get(key, 0) + 1
                    continue

            n_pos = end - start
            fixes, _ = dp_fix_prefix(
                m, x, start, log_probs, mask_id,
                top_k=top_k, end_pos=end, max_positions=n_pos,
                include_masked=True, eos_id=eos_id, eot_id=eot_id,
                objective=objective, cand_source=cand_source, eos_in_candidates=eos_in_candidates,
            )
            if fixes is None:
                stats.failures += 1
                stats.arm_failures[key] = stats.arm_failures.get(key, 0) + 1
                continue

            # gap is a sum over *changed* positions -- unchanged ones cancel --
            # so it stays comparable across arms with different spans, provided
            # both sides are scored on the same positions. Masked positions are
            # the exception and are held out: x has committed nothing there, so
            # repair contributes 0 while the oracle pays a whole token's
            # log-prob, and log-probs are negative. Netting those against each
            # other made a wider arm look worse the more masks it happened to
            # cover -- every negative gap in the pilot (19 of 19) had a mask in
            # span and none without one did. The cost is real, so it is kept,
            # just kept out of the comparison.
            changed = {pos: tok for pos, tok in fixes}
            lp_repair = lp_oracle = 0.0
            mask_fill_nats = 0.0
            filled_masks = 0
            for pos in range(start, end):
                cur = x[0, pos].item()
                tok = changed.get(pos, cur)
                if cur == mask_id:
                    if tok != mask_id:
                        mask_fill_nats += log_probs[0, pos, tok].item()
                        filled_masks += 1
                    continue
                lp_repair += log_probs[0, pos, cur].item()
                lp_oracle += log_probs[0, pos, tok].item()

            record["by_window"][key] = {
                "prefix_edits": sum(1 for pos in changed if pos < violator),
                "total_edits": len(fixes),
                "gap_nats": lp_oracle - lp_repair,
                "mask_fill_nats": mask_fill_nats,
                "filled_masks": filled_masks,
                "span": [start - gen_start, end - gen_start],
            }

        if record["by_window"]:
            stats.sites.append(record)
    except Exception:  # noqa: BLE001 - instrumentation must never break a run
        stats.failures += 1
    finally:
        stats.probe_time += time.perf_counter() - t0


# ── Async mask helper ────────────────────────────────────────────────────────


def _compute_mask_async(checker, vocab_size):
    """Run compute_mask in a background thread.

    Returns (thread, result_holder) where result_holder is a two-element list
    [bias_tensor_or_None, compute_time_seconds].  Call thread.join() before
    reading result_holder.
    """
    result = [None, 0.0]

    def _run():
        t0 = time.perf_counter()
        result[0] = checker.compute_mask(vocab_size=vocab_size)
        result[1] = time.perf_counter() - t0

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result


@torch.no_grad()
# ── Generation loop ──────────────────────────────────────────────────────────


def _extend_prefix(checker, x, consume_idx, mask_id):
    """Consume contiguous non-mask tokens from consume_idx.

    Returns (new_consume_idx, violator_pos_or_-1). Mirrors extend_prefix() in
    generate.py but without the STATS dependency so it can live here.
    """
    tokens = []
    pos = consume_idx
    while pos < x.shape[1]:
        tid = x[0, pos].item()
        if tid == mask_id:
            break
        tokens.append(tid)
        pos += 1
    if not tokens:
        return consume_idx, -1
    count = checker.matcher.try_consume_tokens(tokens)
    if count == len(tokens):
        return consume_idx + count, -1
    return consume_idx + count, consume_idx + count   # violator at consume_idx+count


@torch.no_grad()
def generate_dp(
    model,
    prompt,
    tokenizer,
    checker: TokenChecker,
    prompt_len: int,
    steps: int = 128,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    eos_id: int = 126081,
    eot_id: int = 126348,
    trace: bool = False,
    max_batch_size: int = 8,
    max_resamples: int = 100,
    top_k_dp: int = 100,
    # The forward-scan cap in find_constraint_end.  Exposed so the span budget
    # can be swept without editing the module.
    max_lookahead: int = 48,
    max_dp_secs: float = 300.0,
    stats=None,
    oracle_stats: "OracleStats | None" = None,
    # Parity with top_k_dp. A thinner candidate set than repair's does not make
    # the oracle "conservative": it can drop the very tokens repair chose, so
    # repair's answer leaves the oracle's search space and the arms stop being
    # nested. Set below top_k_dp only to measure that effect deliberately.
    oracle_top_k: int = 100,
    window_mode: str = "progressive",
    # "logp" maximises probability over the span and rewrites positions the
    # grammar never objected to; "min_edit" changes as few as the grammar
    # forces. See dp_fix_prefix.
    objective: str = "logp",
    cand_source: str = "automaton",
    eos_in_candidates: bool = False,
    # Measurement only; costs one integer scan per repair site.
    span_stats: "SpanStats | None" = None,
):
    """Dgrammar with DP-based violation correction + async mask overlap.

    Structurally identical to generate_async_timed: same token placement,
    frontier masking, and inner scheduling loop. Two improvements over the
    original:

      1. Async mask overlap: compute_mask for the frontier token is kicked off
         in a background thread before the forward pass so its CPU cost is
         hidden behind GPU time (same technique as generate_async_timed).

      2. top_k_dp=100: DP explores the top-100 tokens per position instead of
         top-50, increasing the chance of finding a grammar-valid path with
         negligible extra cost (DP is rarely triggered, ~1.2×/sample).

    Falls back to remasking the violator when DP finds no valid path.

    Yields:
        (x, resamples, is_complete, total_violations, total_fixes, total_dp_calls, consume_idx)
    """
    if objective not in ("logp", "min_edit"):
        raise ValueError(f"objective must be 'logp' or 'min_edit'; got {objective!r}")
    if not (window_mode in ("progressive", "full", "descend")
            or (window_mode.startswith("w") and window_mode[1:].isdigit())):
        raise ValueError(
            "window_mode must be 'progressive', 'full', 'descend', or 'wN' "
            "(e.g. 'w8'); "
            f"got {window_mode!r}"
        )

    start_time = time.monotonic()

    def _elapsed() -> float:
        """Wall clock with the oracle probe's cost removed.

        The probe is instrumentation and must not change what the run does, but
        ``max_dp_secs`` and every recorded resample timestamp are measured
        against this clock, so leaving probe time in it lets instrumentation
        truncate the generation it is measuring. It did: raising the probe to
        repair's top_k made it 5x slower, and o10617 then spent 292s of a 300s
        budget with 200s of that in the probe, hit the guard, and produced 5
        repair sites where the identical run without the probe produced 10.
        """
        t = time.monotonic() - start_time
        return t - oracle_stats.probe_time if oracle_stats is not None else t

    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    gen_start = prompt.shape[1]
    consume_idx = gen_start
    current_batch = 1

    if prompt_len < gen_start:
        prefix_tokens = x[0, prompt_len:gen_start].tolist()
        if not checker.consume_tokens(prefix_tokens) and trace:
            print("Warning: prompt suffix rejected by checker")

    total_violations = 0
    total_dp_calls = 0
    total_fixes = 0
    resamples = []

    # Precompute structural bracket token IDs for depth tracking in find_constraint_end.
    # This prevents the DP span from stopping inside an unclosed [ or { (array/object),
    # which would allow the DP to collapse the remainder to [] or {}.
    _open_tok_ids: set = set()
    _close_tok_ids: set = set()
    for _ch, _s in [("[", _open_tok_ids), ("{", _open_tok_ids),
                    ("]", _close_tok_ids), ("}", _close_tok_ids)]:
        _tids = tokenizer.encode(_ch, add_special_tokens=False)
        _s.update(_tids)

    # Grammar matcher state at gen_start (prompt prefix consumed, no generated
    # tokens yet), so the oracle probe can replay to any generated position.
    _initial_matcher = checker.matcher.deep_copy()

    # Pending async mask result: (thread, result_holder) or None.
    # Kicked off just before each forward pass; joined just after.
    pending_mask = None

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            _global_step = num_block * steps_per_block + i
            if complete:
                break

            # ── Skip steps with nothing to place (avoid wasted forward pass) ─
            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            # ── Async mask kick-off (before forward pass) ────────────────────
            # If the frontier is still a mask token and we don't have a pending
            # result, start computing the grammar mask now so it runs in
            # parallel with the GPU forward pass.
            mask_index_pre = x == mask_id
            if (
                pending_mask is None
                and consume_idx < x.shape[1]
                and mask_index_pre[0, consume_idx]
            ):
                vocab_size_hint = 126464  # corrected after logits are available
                pending_mask = _compute_mask_async(checker, vocab_size_hint)

            t_fwd = time.perf_counter()
            logits = model(x).logits
            if stats is not None:
                stats.forward_times.append(time.perf_counter() - t_fwd)
            log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            tokens_placed_this_step = 0
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break
                if _elapsed() > max_dp_secs:
                    if span_stats is not None:
                        span_stats.stopped('dp_budget',
                            at=consume_idx - gen_start,
                            masks_left=int((x[0, gen_start:] == mask_id).sum()))
                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                    return

                # ── Token selection (same as generate_async_timed) ───────────
                mask_index = x == mask_id
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                x0_p[:, block_end:] = -np.inf

                # ── Frontier masking: guarantee first unfilled token is valid ─
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    actual_vocab = logits_with_noise.shape[-1]
                    if pending_mask is not None:
                        # Async result: join (usually already done) and use it.
                        t_wait = time.perf_counter()
                        thread, result_holder = pending_mask
                        thread.join()
                        wait_s = time.perf_counter() - t_wait
                        pending_mask = None
                        if stats is not None:
                            stats.mask_wait_times.append(wait_s)
                            stats.mask_compute_times.append(result_holder[1])
                            stats.overlap_count += 1
                        bias = result_holder[0]
                        # Adjust for vocab size mismatch between hint and actual.
                        if bias.shape[0] > actual_vocab:
                            bias = bias[:actual_vocab]
                        elif bias.shape[0] < actual_vocab:
                            pad = torch.ones(
                                actual_vocab - bias.shape[0], dtype=torch.bool,
                                device=bias.device,
                            )
                            bias = torch.cat([bias, pad])
                    else:
                        # Fallback: synchronous (checker state changed after violation).
                        t_mask = time.perf_counter()
                        bias = checker.compute_mask(vocab_size=actual_vocab)
                        if stats is not None:
                            stats.mask_compute_times.append(time.perf_counter() - t_mask)
                    logits_with_noise[0, consume_idx, bias] = -np.inf
                    x0[0, consume_idx] = torch.argmax(logits_with_noise[0, consume_idx])

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break

                remaining = n_scheduled - tokens_placed_this_step
                batch_k = min(current_batch, remaining, n_available)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)
                if select_indices.shape[0] == 0:
                    if span_stats is not None:
                        span_stats.stopped('nothing_to_place',
                            at=consume_idx - gen_start,
                            masks_left=int((x[0, gen_start:] == mask_id).sum()))
                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                    return

                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vocab_idx = x0[0, pos].item()
                    if logits_with_noise[0, pos, vocab_idx] == -np.inf:
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    if span_stats is not None:
                        span_stats.stopped('nothing_to_place',
                            at=consume_idx - gen_start,
                            masks_left=int((x[0, gen_start:] == mask_id).sum()))
                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                    return

                tokens_placed_this_step += len(positions)
                if stats is not None:
                    stats.tokens_unmasked += len(positions)
                    stats.batch_sizes.append(len(positions))

                # ── Grammar check ────────────────────────────────────────────
                t_gc = time.perf_counter()
                new_idx, violator = _extend_prefix(checker, x, consume_idx, mask_id)
                if stats is not None:
                    stats.grammar_check_times.append(time.perf_counter() - t_gc)

                if violator < 0:
                    consume_idx = new_idx
                    # Only allow batch to grow after 75% of steps have elapsed.
                    # In early steps this keeps current_batch=1, giving the model
                    # one forward pass per committed token so richer context
                    # accumulates before colour/string values are fixed.
                    if _global_step >= steps * 3 // 4:
                        current_batch = min(current_batch * 2, max_batch_size)
                else:
                    total_violations += 1
                    consume_idx = new_idx   # checker is now at the violator position

                    # Checker state is about to change — discard stale async mask.
                    if pending_mask is not None:
                        pending_mask[0].join()
                        pending_mask = None

                    # Two different questions, and they need different tests.
                    #
                    # only_stop_remains: the parser has no consumable
                    # transition left, so ending is not a decision anyone is
                    # making -- it is the only thing that can happen.
                    #
                    # The model placed a stop token at the frontier: the model
                    # has asked to stop and the grammar only has to permit it,
                    # which is exactly what is_accepting() means.  Using
                    # only_stop_remains here would refuse the request whenever
                    # the grammar could also continue -- on SMILES that is every
                    # accepting state (1388 tokens legal), so the model could
                    # never end a molecule.
                    _viol_tok = x[0, violator].item()
                    _osr = only_stop_remains(checker.matcher, eos_id, eot_id)
                    _model_stop = (_viol_tok in (eos_id, eot_id)
                                   and checker.is_accepting())
                    if _osr or _model_stop:
                        if span_stats is not None:
                            span_stats.stopped(
                                "phase0_only_stop" if _osr else "phase0_model_eos",
                                at=violator - gen_start,
                                masks_left=int((x[0, gen_start:] == mask_id).sum()),
                            )
                        if stats is not None:
                            stats.write("eos_fill", eos_id, tokenizer)
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    if span_stats is not None:
                        _b = np.frombuffer(checker.matcher.compute_logit_bias(),
                                           dtype=np.uint8)
                        _legal = np.nonzero(_b)[0]
                        _legal = _legal[(_legal != eos_id) & (_legal != eot_id)]
                        _lp_v = log_probs[0, violator]
                        if _legal.size:
                            _lp_np = _lp_v.detach().to("cpu").numpy()
                            _ord = np.argsort(-_lp_np[_legal])[:6]
                            _top = _legal[_ord].tolist()
                        else:
                            _top = []
                        _dec = lambda t: tokenizer.decode([int(t)])
                        try:
                            _want = _dec(x[0, violator].item())
                            _allow = [_dec(t) for t in _top]
                        except Exception:
                            _want, _allow = "?", []
                        span_stats.violation(
                            want=_want,
                            n_legal=int(_legal.size),
                            allow=_allow,
                        )

                    # ── Phase 1: greedy retry (DGrammar-style, bounded) ──────
                    # Try the top-10 tokens at the violator in logit order.
                    # Failed attempts do NOT count against the global resample
                    # budget — only a successful commit or fallthrough to DP
                    # affects `resamples`.  This prevents a hard violation (valid
                    # token rank > 10) from burning the entire 100-resample
                    # budget before DP ever gets a chance to run.
                    greedy_fixed = False
                    _greedy_attempts = 0
                    _MAX_GREEDY = 10
                    while _greedy_attempts < _MAX_GREEDY:
                        next_vocab = torch.argmax(logits_with_noise[0, violator]).item()
                        if logits_with_noise[0, violator, next_vocab] == -np.inf:
                            break  # exhausted — fall through to DP

                        t_gc_r = time.perf_counter()
                        c_try = checker.matcher.try_consume_tokens([next_vocab])
                        if stats is not None:
                            stats.grammar_check_times.append(time.perf_counter() - t_gc_r)

                        if c_try == 1:
                            x[0, violator] = next_vocab
                            if stats is not None:
                                stats.write("greedy", next_vocab, tokenizer)
                            consume_idx += 1
                            tokens_placed_this_step += 1
                            if stats is not None:
                                stats.tokens_unmasked += 1
                            # greedy resume: consume original tokens that are now valid
                            further_idx, further_viol = _extend_prefix(
                                checker, x, consume_idx, mask_id
                            )
                            consume_idx = further_idx
                            if further_viol >= 0:
                                x[0, consume_idx] = mask_id
                                resamples.append((consume_idx, _elapsed()))
                                if span_stats is not None:
                                    span_stats.resample("single_token")
                                tokens_placed_this_step -= 1
                                if stats is not None:
                                    stats.resample_count += 1
                                    stats.handbacks += 1
                                    stats.tokens_unmasked -= 1
                                if len(resamples) >= max_resamples:
                                    if span_stats is not None:
                                        span_stats.stopped("max_resamples",
                                            at=consume_idx - gen_start,
                                            masks_left=int((x[0, gen_start:] == mask_id).sum()))
                                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                                    return
                            greedy_fixed = True
                            break

                        # token rejected: blacklist it and try the next-best
                        # (not counted as a resample — greedy attempts are free)
                        logits_with_noise[0, violator, next_vocab] = -np.inf
                        if stats is not None:
                            stats.rejections += 1
                        _greedy_attempts += 1

                    if greedy_fixed:
                        current_batch = 1
                        continue

                    # ── Phase 2: DP fallback (only when greedy retry exhausted) ─
                    # Greedy retry ran out of valid tokens in the current logits.
                    # This happens for complex grammar constraints (e.g. a UUID regex)
                    # where no single top-K token at position c is independently
                    # valid — the grammar needs a globally consistent sequence.
                    # DP searches over [c, constraint_end) jointly.
                    _init_depth = 0
                    for _bp in range(gen_start, consume_idx):
                        _btid = x[0, _bp].item()
                        if _btid == mask_id:
                            continue
                        if _btid in _open_tok_ids:
                            _init_depth += 1
                        elif _btid in _close_tok_ids:
                            _init_depth = max(0, _init_depth - 1)

                    _fce_out: dict = {}
                    constraint_end = find_constraint_end(
                        checker.matcher.deep_copy(),
                        x, consume_idx, mask_id,
                        max_lookahead=max_lookahead,
                        open_tok_ids=_open_tok_ids,
                        close_tok_ids=_close_tok_ids,
                        init_depth=_init_depth,
                        out=_fce_out,
                    )

                    # Full segment end (next mask) — used for post-DP greedy resume.
                    seg_end = consume_idx
                    while seg_end < x.shape[1] and x[0, seg_end].item() != mask_id:
                        seg_end += 1

                    total_dp_calls += 1
                    dp_succeeded = False

                    if constraint_end > consume_idx:
                        # Window policy.
                        #
                        # "progressive" doubles the span and stops at the first
                        # size that admits any valid path. That test is weak --
                        # dp_fix_prefix returns non-None whenever *some* token is
                        # grammatical at the violator, which is nearly always
                        # true (`}` closes, `"` opens a key). Measured on a pilot
                        # it settled for width 1 at 24 of 24 repair sites, so the
                        # DP changed a single token, and when that token is a
                        # closer the structure completes and the run pads with
                        # EOS -- which is how `{}` and `[]` get produced.
                        #
                        # "full" hands the whole constraint span to the DP at
                        # once, which is what actually removed the degeneracy:
                        # one-char string leaves fell from 22.3% to 7.3% and
                        # median output length rose from 324 to 466 chars, with
                        # no change to the objective.
                        fixes = None
                        used_end = constraint_end
                        span = constraint_end - consume_idx

                        # Widths to try, in order; the first one that admits a
                        # path wins. The full-span fallback below runs whenever
                        # none does, so every mode fails the same way and the
                        # arms stay comparable.
                        #
                        #   progressive  1, 2, 4, ... -- the published DPGrammar
                        #                order.  Rung 1 parses at nearly every
                        #                site, so the wider rungs never run: 86
                        #                of 103 pilot repairs settled at ws=1 and
                        #                none at 2, 4 or 8.  Kept as the baseline
                        #                arm, not as a default.
                        #   full         one shot at the whole span, itself
                        #                capped at find_constraint_end's
                        #                max_lookahead (48).
                        #   descend      span, span/2, ... 1 -- widest first.
                        #                Both modes above commit to one
                        #                direction and pay for it.  On the n=64
                        #                pilot `progressive` settled at ws=1 on
                        #                86 of its 103 repairs and never once at
                        #                2, 4 or 8, because rung 1 almost always
                        #                parses; `full` is all-or-nothing and
                        #                dead-ended on 42.7% of its DP calls,
                        #                each of which fell through to a remask.
                        #                Narrowing rescues those: width 1 parsed
                        #                at 95.6% of the sites that reached it.
                        #                So take the widest joint solution that
                        #                exists, and only give up the width when
                        #                no path is left.
                        #   wN           one shot at N positions, capped at span.
                        if window_mode == "progressive":
                            widths = []
                            _w = 1
                            while _w < span:
                                widths.append(_w)
                                _w *= 2
                        elif window_mode == "full":
                            widths = []          # straight to the full-span pass
                        elif window_mode == "descend":
                            widths = []
                            _w = span
                            while _w > 1:
                                widths.append(_w)
                                _w //= 2
                            widths.append(1)
                        else:
                            _n = min(int(window_mode[1:]), span)
                            widths = [_n] if _n < span else []

                        # `descend` opens on the full span, so the fallback
                        # below would repeat a DP that already failed.
                        _tried_full = any(
                            consume_idx + ws >= constraint_end for ws in widths
                        )

                        # Records where the last DP attempt died, so the cost
                        # of the all-or-nothing contract is visible.
                        _dp_out: dict = {}

                        for ws in widths:
                            trial_end = consume_idx + ws
                            trial_fixes, trial_reached = dp_fix_prefix(
                                checker.matcher.deep_copy(),
                                x, consume_idx, log_probs, mask_id, top_k=top_k_dp,
                                end_pos=trial_end, eos_id=eos_id, eot_id=eot_id,
                                objective=objective, cand_source=cand_source, eos_in_candidates=eos_in_candidates, out=_dp_out,
                            )
                            if trial_fixes is not None:
                                fixes = trial_fixes
                                used_end = trial_reached
                                break

                        if fixes is None and not _tried_full:  # full window fallback
                            _dp_out = {}
                            fixes, reached = dp_fix_prefix(
                                checker.matcher.deep_copy(),
                                x, consume_idx, log_probs, mask_id, top_k=top_k_dp,
                                end_pos=constraint_end, eos_id=eos_id, eot_id=eot_id,
                                objective=objective, cand_source=cand_source, eos_in_candidates=eos_in_candidates, out=_dp_out,
                            )
                            used_end = reached if fixes is not None else constraint_end

                        if span_stats is not None:
                            # dp_fix_prefix ran with include_masked=False, so
                            # the span it actually optimised ends at the first
                            # MASK, not at used_end.
                            _eff_end = used_end
                            for _p in range(consume_idx, used_end):
                                if x[0, _p].item() == mask_id:
                                    _eff_end = _p
                                    break
                            _fx = dict(fixes) if fixes is not None else {}
                            _d_before = _d_after = _init_depth
                            for _p in range(consume_idx, _eff_end):
                                _t0 = x[0, _p].item()
                                _t1 = _fx.get(_p, _t0)
                                for _t, _key in ((_t0, "b"), (_t1, "a")):
                                    _delta = (
                                        1 if _t in _open_tok_ids
                                        else -1 if _t in _close_tok_ids
                                        else 0
                                    )
                                    if _key == "b":
                                        _d_before = max(0, _d_before + _delta)
                                    else:
                                        _d_after = max(0, _d_after + _delta)
                            span_stats.record(
                                reason=_fce_out.get("reason", "?"),
                                violator=consume_idx - gen_start,
                                constraint_end=constraint_end - gen_start,
                                eff_end=_eff_end - gen_start,
                                span=_eff_end - consume_idx,
                                init_depth=_init_depth,
                                depth_before=_d_before,
                                depth_after=_d_after,
                                n_fixes=len(fixes) if fixes is not None else None,
                                dead_step=_dp_out.get("dead_step"),
                                dp_positions=_dp_out.get("n_positions"),
                                # How big the automaton's edge set actually was,
                                # and how often it fit inside top_k so that the
                                # DP enumerated it exactly instead of sampling.
                                legal_mean=(
                                    _dp_out["legal_total"] / _dp_out["legal_calls"]
                                    if _dp_out.get("legal_calls") else None
                                ),
                                legal_min=_dp_out.get("legal_min"),
                                legal_exact=_dp_out.get("legal_exact", 0),
                                legal_calls=_dp_out.get("legal_calls", 0),
                                bias_disagreements=_dp_out.get("bias_disagreements", 0),
                                eos_in_fixes=any(
                                    t in (eos_id, eot_id) for t in _fx.values()
                                ),
                            )

                        if fixes is not None:
                            for fpos, ftok in fixes:
                                x[0, fpos] = ftok
                                if stats is not None:
                                    stats.write("dp", ftok, tokenizer)
                            total_fixes += len(fixes)
                            if trace and fixes:
                                print(f"  DP fixed {len(fixes)} pos in [{consume_idx-gen_start},{used_end-gen_start}): "
                                      f"{[(p - gen_start, ftok) for p, ftok in fixes]}")

                            # dp_fix_prefix ran with include_masked=False, so it
                            # stopped at the first MASK -- seg_end.  Replaying all
                            # the way to used_end walked a median of 39 positions
                            # past that, into tokens the DP never looked at, and
                            # the first of them is the MASK itself: the replay
                            # could not help failing.  It then "remasked" a
                            # position that was already MASK -- a no-op that still
                            # counted a resample, docked the step's token budget,
                            # and ate the max_resamples allowance.  34 of 43
                            # successful repairs took that path.
                            dp_end = min(used_end, seg_end)
                            dp_tokens = [x[0, p].item() for p in range(consume_idx, dp_end)]
                            c = checker.matcher.try_consume_tokens(dp_tokens)
                            consume_idx += c

                            if c < len(dp_tokens):
                                x[0, consume_idx] = mask_id
                                resamples.append((consume_idx, _elapsed()))
                                if span_stats is not None:
                                    span_stats.resample("dp_span_replay")
                                tokens_placed_this_step -= 1
                                if stats is not None:
                                    stats.resample_count += 1
                                    stats.handbacks += 1
                                    stats.tokens_unmasked -= 1
                                if len(resamples) >= max_resamples:
                                    if span_stats is not None:
                                        span_stats.stopped("max_resamples",
                                            at=consume_idx - gen_start,
                                            masks_left=int((x[0, gen_start:] == mask_id).sum()))
                                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                                    return
                            else:
                                resume_tokens = [x[0, p].item() for p in range(consume_idx, seg_end)]
                                if resume_tokens:
                                    c2 = checker.matcher.try_consume_tokens(resume_tokens)
                                    consume_idx += c2
                                    if c2 < len(resume_tokens):
                                        x[0, consume_idx] = mask_id
                                        resamples.append((consume_idx, _elapsed()))
                                        if span_stats is not None:
                                            span_stats.resample("dp_suffix_replay")
                                        tokens_placed_this_step -= 1
                                        if stats is not None:
                                            stats.resample_count += 1
                                            stats.handbacks += 1
                                            stats.tokens_unmasked -= 1
                                        if len(resamples) >= max_resamples:
                                            if span_stats is not None:
                                                span_stats.stopped('max_resamples',
                                                    at=consume_idx - gen_start,
                                                    masks_left=int((x[0, gen_start:] == mask_id).sum()))
                                            yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                                            return

                            # Instrumentation: what would the same DP have
                            # chosen if the prefix were not frozen? Measurement
                            # only -- runs after the repair is already applied
                            # and touches nothing.
                            if oracle_stats is not None:
                                if fixes:      # nothing repaired, nothing to compare
                                    # Probe forward to `seg_end`, not to the
                                    # `used_end` the progressive window settled
                                    # for. Pinning the oracle to used_end varies
                                    # two things at once -- how far back the
                                    # prefix is released AND how far forward the
                                    # search reaches -- so the W=0 arm would not
                                    # isolate freezing from the ws=1 heuristic.
                                    _oracle_probe(
                                        _initial_matcher, checker.matcher,
                                        x, gen_start, seg_end, used_end,
                                        log_probs, mask_id, violator,
                                        oracle_top_k, oracle_stats,
                                        eos_id=eos_id, eot_id=eot_id,
                                        objective=objective, cand_source=cand_source, eos_in_candidates=eos_in_candidates,
                                    )

                            dp_succeeded = True

                    if not dp_succeeded:
                        x[0, violator] = mask_id
                        resamples.append((violator, _elapsed()))
                        if span_stats is not None:
                            span_stats.resample("dp_dead_end")
                        tokens_placed_this_step -= 1
                        if stats is not None:
                            stats.resample_count += 1
                            stats.handbacks += 1
                            stats.tokens_unmasked -= 1

                        if len(resamples) >= max_resamples:
                            if span_stats is not None:
                                span_stats.stopped('max_resamples',
                                    at=consume_idx - gen_start,
                                    masks_left=int((x[0, gen_start:] == mask_id).sum()))
                            yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                            return

                    current_batch = 1

                # ── Completion checks ────────────────────────────────────────
                if not complete and only_stop_remains(checker.matcher, eos_id, eot_id):
                    gen_ids = x[0, gen_start:].tolist()
                    first_mask = next(
                        (j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids)
                    )
                    if first_mask >= consume_idx - gen_start:
                        if span_stats is not None:
                            span_stats.stopped(
                                "checks_only_stop",
                                at=consume_idx - gen_start,
                                masks_left=int((x[0, gen_start:] == mask_id).sum()),
                            )
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

                # A stop token placed beyond the frontier used to end the
                # document here, filling from its position onward.  It asked
                # the grammar nothing and it kept whatever sat between the
                # frontier and that token -- positions the parser had never
                # validated.  Two of five failures on a 57-instance run came
                # from this: one document was truncated with the parser one
                # token from done, another kept two unparsed tokens after an
                # already-complete object.  A stop token at the frontier is
                # handled above, where the prefix is validated and the grammar
                # can be asked; one beyond it is not yet knowable.

            yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx

        # ── Early block-loop exit ─────────────────────────────────────────────
        # If complete=True was set during this block's step loop, all remaining
        # blocks would only do forward passes with n_scheduled=0 (the sequence
        # is fully filled). Break now to avoid those wasted GPU calls.
        if complete:
            break

    # Clean up any pending async mask thread.
    if pending_mask is not None:
        pending_mask[0].join()
        pending_mask = None

    if span_stats is not None:
        span_stats.stopped("schedule_end",
                           at=consume_idx - gen_start,
                           masks_left=int((x[0, gen_start:] == mask_id).sum()))

    gen_ids = x[0, gen_start:].tolist()
    is_complete = False
    if eos_id in gen_ids or eot_id in gen_ids:
        eos_pos = next(
            (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
        )
        is_complete = eos_pos is not None and mask_id not in gen_ids[:eos_pos]


    yield x, resamples, is_complete, total_violations, total_fixes, total_dp_calls, consume_idx

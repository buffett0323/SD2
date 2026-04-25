"""Run Dgrammar (llguidance) with async overlap: compute_mask runs in parallel with forward pass."""

import json
import time
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
import jsb_dataset  # noqa: F401 - registers jsb_* datasets
from dgrammar.checker import TokenChecker
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens


class TimingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.grammar_check_times = []
        self.token_select_times = []
        self.mask_compute_times = []
        self.mask_wait_times = []  # time waiting for async result after forward
        self.resample_count = 0
        self.tokens_unmasked = 0
        self.batch_sizes = []
        self.overlap_count = 0  # how many mask computes were overlapped

    def summary(self):
        fwd = self.forward_times
        gc = self.grammar_check_times
        ts = self.token_select_times
        mc = self.mask_compute_times
        mw = self.mask_wait_times
        return {
            "forward_count": len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms": (sum(fwd) / len(fwd) * 1000) if fwd else 0,
            "grammar_check_count": len(gc),
            "grammar_check_total_ms": sum(gc) * 1000,
            "grammar_check_mean_ms": (sum(gc) / len(gc) * 1000) if gc else 0,
            "grammar_check_median_ms": (sorted(gc)[len(gc) // 2] * 1000) if gc else 0,
            "grammar_check_p95_ms": (sorted(gc)[int(len(gc) * 0.95)] * 1000) if len(gc) >= 20 else 0,
            "mask_compute_count": len(mc),
            "mask_compute_total_ms": sum(mc) * 1000,
            "mask_compute_mean_ms": (sum(mc) / len(mc) * 1000) if mc else 0,
            "mask_wait_count": len(mw),
            "mask_wait_total_ms": sum(mw) * 1000,
            "mask_wait_mean_ms": (sum(mw) / len(mw) * 1000) if mw else 0,
            "overlap_count": self.overlap_count,
            "token_select_count": len(ts),
            "token_select_total_ms": sum(ts) * 1000,
            "token_select_mean_ms": (sum(ts) / len(ts) * 1000) if ts else 0,
            "resample_count": self.resample_count,
            "tokens_unmasked": self.tokens_unmasked,
            "avg_batch_size": (sum(self.batch_sizes) / len(self.batch_sizes)) if self.batch_sizes else 0,
        }


STATS = TimingStats()


def compute_mask_async(checker, vocab_size):
    """Run compute_mask in a background thread, return (result, compute_time)."""
    result = [None, 0.0]

    def _run():
        t0 = time.perf_counter()
        result[0] = checker.compute_mask(vocab_size=vocab_size)
        result[1] = time.perf_counter() - t0

    thread = threading.Thread(target=_run)
    thread.start()
    return thread, result


def extend_prefix_timed(checker, x, consume_idx, mask_id):
    """extend_prefix with timing."""
    t0 = time.perf_counter()
    tokens_to_consume = []
    pos = consume_idx
    while pos < x.shape[1]:
        tid = x[0, pos].item()
        if tid == mask_id:
            break
        tokens_to_consume.append(tid)
        pos += 1

    if not tokens_to_consume:
        STATS.grammar_check_times.append(time.perf_counter() - t0)
        return consume_idx, -1

    count = checker.matcher.try_consume_tokens(tokens_to_consume)
    STATS.grammar_check_times.append(time.perf_counter() - t0)

    if count == len(tokens_to_consume):
        return consume_idx + count, -1
    else:
        return consume_idx + count, consume_idx + count


def _pattern_min_string(pattern: str) -> str:
    """Generate a minimal string satisfying a simple regex pattern.

    Handles \\d, \\w, \\s, [char-class], {N}, {N,M}, ?, *, +, (, |, ., literal chars.
    Used by _minimal_json_value for schemas with 'pattern' constraints.
    """
    p = pattern.lstrip('^').rstrip('$')
    result: list[str] = []
    i = 0
    while i < len(p):
        c = p[i]
        if c == '\\' and i + 1 < len(p):
            nc = p[i + 1]
            if nc == 'd':   result.append('0')
            elif nc == 'w': result.append('a')
            elif nc == 's': result.append(' ')
            elif nc == 'D': result.append('a')
            elif nc == 'W': result.append(' ')
            elif nc == 'S': result.append('a')
            else:           result.append(nc)
            i += 2
        elif c == '[':
            j = p.find(']', i + 1)
            if j < 0: j = len(p) - 1
            cls = p[i + 1:j]
            if cls.startswith('^'): cls = cls[1:]
            ch = cls[0] if cls else 'a'  # first char in class
            result.append(ch)
            i = j + 1
            # consume quantifier
            if i < len(p) and p[i] == '{':
                j2 = p.find('}', i + 1)
                if j2 >= 0:
                    try:
                        n = int(p[i + 1:j2].split(',')[0])
                        result[-1:] = [ch] * n
                    except ValueError:
                        pass
                    i = j2 + 1
                else:
                    i += 1
            elif i < len(p) and p[i] == '+': i += 1
            elif i < len(p) and p[i] == '*': result.pop(); i += 1
            elif i < len(p) and p[i] == '?': result.pop(); i += 1  # use 0 occurrences
        elif c == '(':
            # find matching ')'
            depth_g, j = 1, i + 1
            while j < len(p) and depth_g > 0:
                if p[j] == '(': depth_g += 1
                elif p[j] == ')': depth_g -= 1
                j += 1
            inner = p[i + 1:j - 1]
            first_alt = inner.split('|')[0]
            sub = _pattern_min_string(first_alt)
            result.append(sub)
            i = j
            if i < len(p) and p[i] == '?': result.pop(); i += 1  # optional → skip
            elif i < len(p) and p[i] == '*': result.pop(); i += 1
            elif i < len(p) and p[i] == '+': i += 1
            elif i < len(p) and p[i] == '{':
                j2 = p.find('}', i + 1)
                if j2 >= 0:
                    try:
                        n = max(1, int(p[i + 1:j2].split(',')[0]))
                        last = result.pop() if result else sub
                        result.extend([last] * n)
                    except ValueError:
                        pass
                    i = j2 + 1
                else:
                    i += 1
        elif c == '{':
            j = p.find('}', i + 1)
            if j >= 0:
                try:
                    n = max(1, int(p[i + 1:j].split(',')[0]))
                    if result:
                        last = result.pop()
                        result.extend([last] * n)
                except ValueError:
                    pass
                i = j + 1
            else:
                result.append(c); i += 1
        elif c == '.': result.append('x'); i += 1
        elif c == '|': break  # use first alternative only
        elif c in '+*?':
            if c in '*?' and result: result.pop()
            i += 1
        else:
            result.append(c); i += 1
    return ''.join(result)


def _minimal_json_value(schema, depth=0, root_schema=None):
    """Return a minimal Python value (JSON-serialisable) satisfying `schema`.

    Handles common JSON Schema keywords including:
    - pattern: uses _pattern_min_string to generate a matching string
    - $ref: resolved against root_schema["definitions"]
    - properties ordering: uses schema definition order (typically alphabetical),
      not required-list order, to match llguidance grammar enforcement order
    """
    if depth > 10 or not isinstance(schema, dict):
        return ""

    # Resolve $ref first
    if "$ref" in schema and root_schema is not None:
        ref = schema["$ref"]
        if ref.startswith("#/definitions/"):
            def_name = ref[len("#/definitions/"):]
            defs = root_schema.get("definitions", {})
            if def_name in defs:
                merged = dict(defs[def_name])
                merged.update({k: v for k, v in schema.items() if k != "$ref"})
                return _minimal_json_value(merged, depth, root_schema)

    # const / enum take absolute priority
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]

    # anyOf / oneOf / allOf — try first sub-schema
    for kw in ("anyOf", "oneOf", "allOf"):
        if kw in schema and schema[kw]:
            merged = dict(schema)
            merged.update(schema[kw][0])
            merged.pop(kw, None)
            val = _minimal_json_value(merged, depth, root_schema)
            if val is not None:
                return val

    type_ = schema.get("type")
    if isinstance(type_, list):
        for t in type_:
            if t == "null":
                continue  # prefer non-null
            v = _minimal_json_value(dict(schema, type=t), depth, root_schema)
            if v is not None:
                return v
        return None

    if type_ == "null":
        return None
    if type_ == "boolean":
        return True
    if type_ in ("integer", "number"):
        mn = schema.get("minimum", schema.get("exclusiveMinimum", -1))
        if mn is not None and mn > 0:
            val = int(mn) + (1 if schema.get("exclusiveMinimum") == mn else 0)
        else:
            val = 0
        mx = schema.get("maximum", schema.get("exclusiveMaximum"))
        if mx is not None and val > mx:
            val = int(mx)
        return val
    if type_ == "string":
        fmt = schema.get("format", "")
        pattern = schema.get("pattern", "")
        min_len = schema.get("minLength", 0)
        if fmt in ("uri", "iri", "uri-reference", "iri-reference"):
            return "http://x.example.com"
        if fmt in ("date-time", "datetime"):
            return "2000-01-01T00:00:00Z"
        if fmt == "date":
            return "2000-01-01"
        if fmt == "time":
            return "00:00:00Z"
        if fmt == "email":
            return "x@x.com"
        if fmt == "uuid":
            return "00000000-0000-0000-0000-000000000000"
        if fmt == "hostname":
            return "x.example.com"
        if fmt == "ipv4":
            return "0.0.0.0"
        if fmt == "ipv6":
            return "::1"
        # Pattern constraint: generate a minimal matching string
        if pattern:
            s = _pattern_min_string(pattern)
            if len(s) >= max(1, min_len):
                return s
        return "x" * max(1, min_len)
    if type_ == "array":
        min_items = schema.get("minItems", 0)
        items_schema = schema.get("items", {})
        if not isinstance(items_schema, dict):
            items_schema = {}
        return [_minimal_json_value(items_schema, depth + 1, root_schema)
                for _ in range(max(0, min_items))]

    # object (explicit or implied by presence of properties/required)
    if type_ == "object" or "properties" in schema or "required" in schema:
        required_set = set(schema.get("required", []))
        properties = schema.get("properties", {})
        result = {}
        # Use properties dict ORDER (typically alphabetical = grammar enforcement order)
        # NOT the required-list order, which may differ from grammar's expected order.
        # llguidance enforces properties in definition/alphabetical order; using required-
        # list order causes guided_encode to fail with n_valid=1 at property name positions.
        for key in properties:
            if key in required_set:
                prop_schema = properties[key]
                val = _minimal_json_value(prop_schema, depth + 1, root_schema)
                result[key] = val if val is not None else ""
        # Include any required properties not in properties dict (fallback)
        for key in schema.get("required", []):
            if key not in result:
                result[key] = ""
        return result

    # No explicit type — check format/pattern for type hints
    fmt = schema.get("format", "")
    pattern = schema.get("pattern", "")
    if fmt in ("uri", "iri", "uri-reference", "iri-reference"):
        return "http://x.example.com"
    if fmt in ("date-time", "datetime"):
        return "2000-01-01T00:00:00Z"
    if fmt == "date":
        return "2000-01-01"
    if fmt == "time":
        return "00:00:00Z"
    if fmt == "email":
        return "x@x.com"
    if fmt == "uuid":
        return "00000000-0000-0000-0000-000000000000"
    if fmt == "hostname":
        return "x.example.com"
    if fmt == "ipv4":
        return "0.0.0.0"
    if fmt == "ipv6":
        return "::1"
    # Default to empty string
    return ""


_STRUCTURAL_CHARS = frozenset({'"', '}', ']', ',', ':'})
# SentencePiece uses U+2581 (▁) as a word-leading space marker.
_SP_PREFIX = '\u2581'


def _decode_stripped(tokenizer, tid):
    """Decode a single token, stripping SentencePiece space prefix if present."""
    s = tokenizer.decode([tid], skip_special_tokens=False)
    return s.lstrip(_SP_PREFIX + ' ')


def _grammar_guided_encode(checker_clone, target_str, tokenizer, vocab_size, max_steps=400):
    """Encode target_str into a grammar-valid token sequence.

    Unlike tokenizer.encode(), tokens are chosen via compute_mask at each step,
    avoiding SentencePiece space-prefix artifacts (▁resource, ▁lis, etc.) that
    would be rejected by the JSON grammar inside a string context.

    Fast path: try the tokenizer's natural tokenization of the remaining string
    first. If the first token is grammar-valid, use it. Otherwise, scan all valid
    tokens for one whose raw bytes match the target at the current position.

    Returns list of token IDs, or None if target_str cannot be encoded.
    """
    target_bytes = target_str.encode('utf-8')
    byte_pos = 0
    token_ids = []

    for _ in range(max_steps):
        if byte_pos >= len(target_bytes):
            return token_ids if checker_clone.is_accepting() else None

        remaining = target_bytes[byte_pos:]
        mask = checker_clone.compute_mask(vocab_size=vocab_size)

        # Fast path: try tokenizer's natural first token for remaining string.
        try:
            remaining_str = remaining.decode('utf-8', errors='replace')
            cand_ids = tokenizer.encode(remaining_str, add_special_tokens=False)
        except Exception:
            cand_ids = []

        chosen_tid = None
        chosen_len = 0

        if cand_ids:
            first_tid = cand_ids[0]
            if not mask[first_tid]:  # grammar accepts
                raw_bytes = tokenizer.decode([first_tid], skip_special_tokens=False).encode('utf-8')
                if len(raw_bytes) > 0 and remaining[:len(raw_bytes)] == raw_bytes:
                    chosen_tid = first_tid
                    chosen_len = len(raw_bytes)

        # Medium path: X-prefix trick — tokenize "X"+remaining to get non-▁ tokens.
        # SentencePiece attaches ▁ to word-initial characters; prepending "X" forces
        # word-internal tokenization for the remaining string.
        if chosen_tid is None:
            try:
                remaining_str = remaining.decode('utf-8', errors='replace')
                xpfx_ids = tokenizer.encode("X" + remaining_str, add_special_tokens=False)
            except Exception:
                xpfx_ids = []
            if len(xpfx_ids) > 1:
                tid = xpfx_ids[1]  # first token of remaining (non-▁)
                if not mask[tid]:
                    raw_bytes = tokenizer.decode([tid], skip_special_tokens=False).encode('utf-8')
                    n = len(raw_bytes)
                    if n > 0 and n <= len(remaining) and remaining[:n] == raw_bytes:
                        chosen_tid = tid
                        chosen_len = n

        # Slow path: scan valid tokens for one whose bytes match target.
        if chosen_tid is None:
            valid_ids = (~mask).nonzero(as_tuple=True)[0].tolist()
            for tid in valid_ids:
                raw_bytes = tokenizer.decode([tid], skip_special_tokens=False).encode('utf-8')
                n = len(raw_bytes)
                if n > 0 and n <= len(remaining) and remaining[:n] == raw_bytes and n > chosen_len:
                    chosen_len = n
                    chosen_tid = tid

        if chosen_tid is None:
            print(f"    [guided_encode] stuck at byte {byte_pos}: {remaining[:20]!r}")
            # Debug: show first few valid tokens at this position
            _dbg_valid = (~mask).nonzero(as_tuple=True)[0].tolist()
            print(f"      n_valid={len(_dbg_valid)}")
            for _dbg_tid in _dbg_valid[:8]:
                _dbg_rb = tokenizer.decode([_dbg_tid], skip_special_tokens=False).encode('utf-8')
                print(f"      valid[{_dbg_tid}]={_dbg_rb!r}")
            return None

        c = checker_clone.matcher.try_consume_tokens([chosen_tid])
        if c != 1:
            return None
        token_ids.append(chosen_tid)
        byte_pos += chosen_len

    return None


def _force_close_grammar(checker, vocab_size, max_steps=2048, priority_ids=None, tokenizer=None, deadline=None):
    """Greedy grammar-only walk to is_accepting().

    Uses two-tier scoring so structural JSON tokens (`"`, `}`, `]`, `,`, `:`)
    ALWAYS beat content tokens (e.g. `\\t`, `\\n` inside a string):

      Tier 0  (is_accepting immediately) → score = 0
      Tier 1  (structural token)         → score = next_count        [always < tier 2]
      Tier 2  (non-structural token)     → score = next_count + 2*(vocab_size+1)

    This prevents cycling where content tokens have lower raw next_count than
    structural closing tokens (e.g. `\\t` stays in 94-token string state while
    `"` moves to a 500-token object state).

    When valid set ≤ 30: evaluate ALL valid tokens.
    When large: evaluate runtime-decoded structural tokens + priority_ids + first 6 low-ID tokens.

    Returns list of token IDs, or None if not accepting after max_steps.
    Checker state is left unchanged (all consumed tokens are rolled back).
    """
    if checker.is_accepting():
        return []

    priority_ids = set(priority_ids or [])
    sequence = []
    STRUCT_TIER_OFFSET = 0
    CONTENT_TIER_OFFSET = 2 * (vocab_size + 1)

    # closing_ids_set: only }, ] and EOS/EOT can trigger is_accepting().
    # In JSON, only closing brackets and end-of-sequence markers complete a
    # value.  Calling is_accepting() for ", ,, : costs 500ms+ per call on
    # large NFA state spaces: a JSON object with 10 properties would have
    # 20+ such calls (500ms × 20 = 10s wasted).  Restricting to }, ] and
    # EOS/EOT limits calls to O(nesting depth) per force_close run.
    _CLOSING_CHARS = {'}', ']'}
    if tokenizer is not None:
        closing_ids_set = {tid for tid in priority_ids
                           if _decode_stripped(tokenizer, tid) in _CLOSING_CHARS}
    else:
        closing_ids_set = set(priority_ids)
    closing_ids_set |= {126081, 126348}  # EOS, EOT (LLaDA-8B-Instruct)

    for _step in range(max_steps):
        if deadline is not None and time.monotonic() > deadline:
            break
        # is_accepting() NOT called here — too expensive (500ms on large NFAs).
        # Instead we rely on the final check after the loop (line below).
        bias = checker.compute_mask(vocab_size=vocab_size)
        valid_ids = (~bias).nonzero(as_tuple=True)[0].tolist()
        if not valid_ids:
            break

        valid_set = set(valid_ids)

        # REMOVED: per-step O(n_valid × decode_time) runtime_structural scan.
        # Previously iterated all valid tokens and called _decode_stripped on
        # each to find structural chars.  For n_valid=1000+, this was
        # 1000 × 0.1ms = 100ms per step → force_close timeout on fresh checker.
        # priority_ids already covers all structural chars (", }, ], ,, :) and
        # is O(5) to intersect with valid_set.
        structural_set = priority_ids & valid_set

        # Build candidate pool: O(|priority_ids| + 6) for large valid sets.
        if len(valid_ids) <= 30:
            candidates = list(valid_ids)
        else:
            candidates = [p for p in priority_ids if p in valid_set]
            cand_set = set(candidates)
            for tid in valid_ids[:6]:
                if tid not in cand_set:
                    candidates.append(tid)

        chosen = valid_ids[0]
        best_score = float("inf")

        for candidate in candidates:
            c = checker.matcher.try_consume_tokens([candidate])
            if c != 1:
                continue
            # is_accepting() only for closing tokens (}, ], EOS, priority_ids).
            # In JSON, only these can finalize the grammar.  Calling it for
            # every candidate (e.g. letter tokens while inside a property name)
            # wastes 500ms × N_candidates × 30 steps = 7-45s per force_close.
            if candidate in closing_ids_set and checker.is_accepting():
                checker.matcher.rollback(1)
                chosen = candidate
                best_score = -1.0
                break
            bias2 = checker.compute_mask(vocab_size=vocab_size)
            next_count = int((~bias2).sum().item())
            checker.matcher.rollback(1)

            # Skip dead-end candidates: if next_count==0, the grammar has no
            # valid continuation after this token (e.g. `"` inside a format-
            # constrained string that hasn't satisfied the format yet).
            # Choosing such a token would permanently block force_close.
            if next_count == 0:
                continue

            # Two-tier: structural tokens always beat non-structural.
            offset = STRUCT_TIER_OFFSET if candidate in structural_set else CONTENT_TIER_OFFSET
            score = next_count + offset
            if score < best_score:
                best_score = score
                chosen = candidate

        c = checker.matcher.try_consume_tokens([chosen])
        if c != 1:
            break
        sequence.append(chosen)
        # Early exit: if we just consumed a closing token, check acceptance.
        if chosen in closing_ids_set and checker.is_accepting():
            break

    success = checker.is_accepting()
    checker.matcher.rollback(len(sequence))
    return sequence if success else None


@torch.no_grad()
def autocomplete_greedy(model, x, checker, consume_idx, gen_start, mask_id, eos_id,
                        refresh_interval=8, closing_bonus=0.0, max_steps=256,
                        deadline: float | None = None,
                        closing_token_ids: set | None = None):
    """Grammar-guided greedy completion from consume_idx forward.

    Diffusion models give logits for all positions at once. We exploit this by
    doing one forward pass and completing multiple positions from the same logits.
    Logits are refreshed every `refresh_interval` steps to avoid staleness.

    closing_bonus: when > 0, a 2-step lookahead checks which valid tokens
        lead to is_accepting() within 1–2 steps and adds closing_bonus (step 1)
        or closing_bonus*0.5 (step 2) to their logit before argmax.  Use a
        large value (e.g. 100) in the budget-exhaustion extension pass so the
        model prefers closing brackets/braces over generating more content.

    max_steps: hard cap on ac_steps to avoid runaway loops on complex schemas.
        When reached, return early so pass 3a/3b can handle the remainder.

    deadline: if set (time.monotonic() timestamp), exit early when exceeded.

    Returns (x, autocomplete_steps, autocomplete_mask_ms, autocomplete_fwd_ms).
    """
    ac_steps = 0
    ac_fwd_ms = 0.0
    ac_mask_ms = 0.0
    ac_tc_ms = 0.0   # try_consume time (not previously measured)
    seq_len = x.shape[1]
    steps_since_refresh = refresh_interval  # force initial forward

    while consume_idx < seq_len and ac_steps < max_steps:
        if deadline is not None and time.monotonic() > deadline:
            break

        # Refresh logits periodically.
        # is_accepting() is NOT called here — costs 15-170ms per call on
        # large NFA state spaces, and 64 refresh-boundary calls per 512-step
        # run = 5-11s of pure overhead.  Early acceptance is handled naturally:
        # closing_bonus boosts grammar-valid }, ] and EOS (closing_token_ids),
        # so once the grammar accepts, those tokens win the argmax and the
        # dead-end check below fires one final is_accepting() to fill EOS.
        if steps_since_refresh >= refresh_interval:
            t_fwd = time.perf_counter()
            logits = model(x).logits
            ac_fwd_ms += (time.perf_counter() - t_fwd) * 1000
            steps_since_refresh = 0

        # compute_mask at frontier
        t_mc = time.perf_counter()
        bias = checker.compute_mask(vocab_size=logits.shape[-1])
        ac_mask_ms += (time.perf_counter() - t_mc) * 1000

        # Move one logit row to CPU before masking and argmax.
        # GPU in-place scatter (logits[bias]=-inf) + GPU→CPU sync in .item()
        # costs ~10ms/step; CPU-side ops are <0.1ms.  This is a global fix
        # (benefits all autocomplete_greedy callers, not just hard instances).
        row = logits[0, consume_idx].cpu().float()
        row[bias] = float('-inf')

        # Boost grammar-valid closing tokens when budget is exhausted.
        # Grammar mask already guarantees }, ] are only valid when all required
        # fields are satisfied, so no is_accepting() probe is needed.
        # Old approach (try_consume + is_accepting() per valid token) cost
        # 15-40ms per is_accepting() × n_valid × 512 steps = 5-20s of waste.
        # New approach: O(len(closing_token_ids)) per step, zero is_accepting()
        # calls.  closing_token_ids should be pre-computed token IDs for }, ],
        # and EOS/EOT — passed in by the caller that sets closing_bonus.
        if closing_bonus > 0.0 and closing_token_ids:
            for tid in closing_token_ids:
                if tid < row.shape[0] and row[tid].item() != float('-inf'):
                    row[tid] += closing_bonus

        best = int(row.argmax())

        if row[best].item() == float('-inf'):
            # Dead-end: check one final time before giving up
            if checker.is_accepting():
                for j in range(consume_idx, seq_len):
                    x[0, j] = eos_id
            break

        x[0, consume_idx] = best

        _t_tc = time.perf_counter()
        c = checker.matcher.try_consume_tokens([best])
        ac_tc_ms += (time.perf_counter() - _t_tc) * 1000
        if c == 1:
            consume_idx += 1
            ac_steps += 1
            steps_since_refresh += 1
        else:
            break

        # Extend past already-placed non-mask tokens
        while consume_idx < seq_len:
            tid = x[0, consume_idx].item()
            if tid == mask_id:
                break
            c = checker.matcher.try_consume_tokens([tid])
            if c == 1:
                consume_idx += 1
                ac_steps += 1
                steps_since_refresh += 1
            else:
                x[0, consume_idx] = mask_id
                break

    return x, ac_steps, ac_mask_ms, ac_fwd_ms, consume_idx, ac_tc_ms


@torch.no_grad()
def generate_async_timed(
    model, prompt, tokenizer, checker,
    prompt_len, steps=128, gen_length=256,
    block_length=32, temperature=0.0,
    remasking="low_confidence", mask_id=126336,
    eos_id=126081, eot_id=126348,
    max_batch_size=8, max_resamples=100,
):
    start_time = time.monotonic()

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    total_violations = 0
    total_remasks = 0
    total_grammar_checks = 0
    resamples = []
    current_batch = 1

    gen_start = prompt.shape[1]
    consume_idx = gen_start

    if prompt_len < gen_start:
        prefix_tokens = x[0, prompt_len:gen_start].tolist()
        checker.consume_tokens(prefix_tokens)

    # Pending async mask result from previous iteration
    pending_mask = None  # (thread, result_list, vocab_size)

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            # --- Check if we should precompute mask async BEFORE forward ---
            # Grammar state is known now. If frontier is a mask, start compute_mask.
            mask_index_pre = x == mask_id
            need_mask = (
                consume_idx < x.shape[1]
                and mask_index_pre[0, consume_idx]
                and pending_mask is None
            )
            if need_mask:
                vocab_size = 126464  # will be corrected after logits are available
                thread, result_holder = compute_mask_async(checker, vocab_size)
                pending_mask = (thread, result_holder)

            # --- Model forward (timed) ---
            t_fwd = time.perf_counter()
            logits = model(x).logits
            STATS.forward_times.append(time.perf_counter() - t_fwd)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed_this_step = 0
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

                # --- Token selection (timed) ---
                t_sel = time.perf_counter()
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

                # Frontier masking: use async result if available
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    if pending_mask is not None:
                        # Wait for async result
                        t_wait = time.perf_counter()
                        thread, result_holder = pending_mask
                        thread.join()
                        wait_time = time.perf_counter() - t_wait
                        STATS.mask_wait_times.append(wait_time)
                        STATS.mask_compute_times.append(result_holder[1])
                        STATS.overlap_count += 1
                        bias = result_holder[0]
                        pending_mask = None
                        # Truncate/pad if vocab sizes differ
                        actual_vocab = logits_with_noise.shape[-1]
                        if bias.shape[0] > actual_vocab:
                            bias = bias[:actual_vocab]
                        elif bias.shape[0] < actual_vocab:
                            pad = torch.ones(actual_vocab - bias.shape[0], dtype=torch.bool)
                            bias = torch.cat([bias, pad])
                    else:
                        # Fallback: compute synchronously (e.g. after resampling changed state)
                        t_mask = time.perf_counter()
                        bias = checker.compute_mask(vocab_size=logits_with_noise.shape[-1])
                        STATS.mask_compute_times.append(time.perf_counter() - t_mask)

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
                STATS.token_select_times.append(time.perf_counter() - t_sel)
                STATS.batch_sizes.append(batch_k)

                if select_indices.shape[0] == 0:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
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
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                tokens_placed_this_step += len(positions)
                STATS.tokens_unmasked += len(positions)

                # --- Grammar check (timed) ---
                total_grammar_checks += 1
                new_idx, violator = extend_prefix_timed(checker, x, consume_idx, mask_id)

                if violator < 0:
                    consume_idx = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                else:
                    total_violations += 1
                    consume_idx = new_idx

                    if checker.is_accepting():
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    bad_token = x[0, violator].item()
                    logits_with_noise[0, violator, bad_token] = -np.inf
                    x[0, violator] = mask_id
                    total_remasks += 1
                    STATS.resample_count += 1
                    tokens_placed_this_step -= 1
                    STATS.tokens_unmasked -= 1
                    resamples.append((violator, time.monotonic() - start_time))

                    if len(resamples) >= max_resamples:
                        yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                        return

                    found = False
                    while len(resamples) < max_resamples:
                        next_vocab = torch.argmax(logits_with_noise[0, violator]).item()
                        if logits_with_noise[0, violator, next_vocab] == -np.inf:
                            break
                        t_gc_retry = time.perf_counter()
                        total_grammar_checks += 1
                        c = checker.matcher.try_consume_tokens([next_vocab])
                        STATS.grammar_check_times.append(time.perf_counter() - t_gc_retry)
                        if c == 1:
                            x[0, violator] = next_vocab
                            consume_idx += 1
                            tokens_placed_this_step += 1
                            STATS.tokens_unmasked += 1
                            found = True
                            further_idx, further_viol = extend_prefix_timed(
                                checker, x, consume_idx, mask_id
                            )
                            total_grammar_checks += 1
                            if further_viol < 0:
                                consume_idx = further_idx
                            else:
                                consume_idx = further_idx
                            break
                        logits_with_noise[0, violator, next_vocab] = -np.inf
                        total_remasks += 1
                        STATS.resample_count += 1
                        resamples.append((violator, time.monotonic() - start_time))
                    current_batch = 1

                # Check completion
                if not complete and checker.is_accepting():
                    gen_ids = x[0, gen_start:].tolist()
                    first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids))
                    if first_mask >= consume_idx - gen_start:
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

                if not complete:
                    gen_ids = x[0, gen_start:].tolist()
                    if eos_id in gen_ids or eot_id in gen_ids:
                        eos_pos = next((j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None)
                        if eos_pos is not None and mask_id not in gen_ids[:eos_pos]:
                            for j in range(eos_pos, len(gen_ids)):
                                x[0, gen_start + j] = x[0, gen_start + eos_pos]
                            complete = True

            yield x, resamples, False, total_violations, total_remasks, total_grammar_checks

    # Clean up any pending async
    if pending_mask is not None:
        thread, _ = pending_mask
        thread.join()
        pending_mask = None

    gen_ids = x[0, gen_start:].tolist()
    is_complete = False
    if eos_id in gen_ids or eot_id in gen_ids:
        eos_pos = next((j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None)
        is_complete = eos_pos is not None and mask_id not in gen_ids[:eos_pos]
    yield x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    block_ar = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    method = sys.argv[7] if len(sys.argv) > 7 else "dgrammar"
    # Optional: comma-separated instance IDs to run only those instances
    # e.g. "o33928,o12618,o70379"
    instance_ids_filter: set | None = None
    if len(sys.argv) > 8 and sys.argv[8]:
        instance_ids_filter = set(sys.argv[8].split(","))
    deviation_penalty = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0

    if method == "dp":
        tag = "dp"
    elif not block_ar:
        tag = "v2_async_ac4_fullpar_timed"
    else:
        tag = "v2_async_ac4_timed"
    ds_safe = dataset_name.replace("/", "_")
    sfx = f"_off{offset}" if offset > 0 else ""
    output_file = f"results/{tag}_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"

    if method == "dp":
        from dgrammar.dp_generate import generate_dp

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    if instance_ids_filter is not None:
        instances = [inst for inst in all_instances if inst.instance_id() in instance_ids_filter]
    else:
        instances = all_instances[offset:offset + limit]
    bl = 32 if block_ar else 256
    print(f"Dgrammar timed [{method}]: {len(instances)} instances, seed={seed}, T={steps}, block_length={bl}")

    cached_checker = None

    # Precompute structural JSON token IDs so _force_close_grammar always
    # evaluates them regardless of their position in sorted valid_ids.
    # `"` typically has a high token ID in Mistral-family tokenizers (e.g. 28739)
    # and would never appear in valid_ids[:6], causing force_close to cycle on `\n`.
    _fc_priority_ids = set()
    for _ch in ['"', '}', ']', ',', ':']:
        _tids = tokenizer.encode(_ch, add_special_tokens=False)
        if _tids:
            _fc_priority_ids.add(_tids[-1])

    # Closing tokens for closing_bonus in pass3b: }, ] and EOS/EOT.
    # Grammar-valid }, ] already guarantee all required JSON fields are
    # satisfied, so no is_accepting() probe is needed before boosting.
    # EOS=126081, EOT=126348 are fixed for LLaDA-8B-Instruct tokenizer.
    _fc_closing_ids: set[int] = {126081, 126348}
    for _ch in ['}', ']']:
        _tids = tokenizer.encode(_ch, add_special_tokens=False)
        if _tids:
            _fc_closing_ids.add(_tids[-1])

    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            print(f"  Skipping {instance.instance_id()}: no schema")
            continue

        try:
            if cached_checker is None or dataset.different_grammar_per_instance:
                cached_checker = TokenChecker(schema_str)
            checker = cached_checker.clone()
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: {e}")
            continue

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        print(f"[{i+1}/{len(instances)}] {instance.instance_id()} ...")
        STATS.reset()
        torch.manual_seed(seed)
        start_time = time.monotonic()

        out = None
        resamples = []
        valid = False
        total_violations = 0
        total_remasks = 0
        total_grammar_checks = 0

        ac_steps = 0
        ac_mask_ms = 0.0
        ac_fwd_ms = 0.0
        dp_consume_idx = None  # final consume_idx from DP generator

        if method == "dp":
            gen_kwargs = dict(stats=STATS)
            gen_fn = generate_dp
        else:
            gen_kwargs = {}
            gen_fn = generate_async_timed

        if method == "dp":
            _gen_t0 = time.monotonic()
            for out, resamples, valid, violations, remasks, grammar_checks, dp_consume_idx in gen_fn(
                model, prompt_ids, tokenizer, checker=checker,
                prompt_len=prompt_len, steps=steps, gen_length=256,
                block_length=bl, temperature=0.2, remasking="low_confidence",
                max_batch_size=8, max_resamples=100, max_dp_secs=240.0,
                deviation_penalty=deviation_penalty, **gen_kwargs,
            ):
                total_violations = violations
                total_remasks = remasks
                total_grammar_checks = grammar_checks
            print(f"  [dp] gen done in {time.monotonic()-_gen_t0:.1f}s  violations={total_violations} resamples={len(resamples)} dp_calls={grammar_checks}")
        else:
            for out, resamples, valid, violations, remasks, grammar_checks in gen_fn(
                model, prompt_ids, tokenizer, checker=checker,
                prompt_len=prompt_len, steps=steps, gen_length=256,
                block_length=bl, temperature=0.2, remasking="low_confidence",
                max_batch_size=8, max_resamples=100, **gen_kwargs,
            ):
                total_violations = violations
                total_remasks = remasks
                total_grammar_checks = grammar_checks

        # ── Per-instance wall-clock budget: 270s for gen, 60s for autocomplete,
        #    leaving headroom for force_close and I/O within a 360s total budget.
        _INSTANCE_BUDGET = 360.0  # seconds
        _instance_deadline = start_time + _INSTANCE_BUDGET

        # ── Autocompletion fallback ───────────────────────────────────────────
        # Strategy (applies to both dgrammar and DP):
        #   Pass 1 — run autocomplete_greedy from the grammar-tracked frontier.
        #            This handles two sub-cases:
        #              (a) Remaining masks in the sequence (early-completed samples).
        #              (b) Already-placed tokens after consume_idx that are
        #                  grammar-invalid; autocomplete_greedy remasks and
        #                  replaces them token-by-token.
        #   Pass 2 — if still invalid AND the checker is not accepting AND the
        #            sequence has no masks left (budget exhausted), extend by
        #            EXTENSION_LEN masked tokens and run autocomplete again.
        #            This completes truncated JSONs without paying extra cost for
        #            the 80%+ of samples that finish within the original budget.
        EXTENSION_CHUNK = 128   # tokens added per extension round
        MAX_EXTENSIONS  = 3     # up to 3 × 128 = 384 extra tokens total
        mask_id_val   = 126336
        eos_id_val    = 126081
        eot_id_val    = 126348

        def _recheck_valid(tensor, gs):
            ids = tensor[0, gs:].tolist()
            if eos_id_val in ids or eot_id_val in ids:
                ep = next((j for j, t in enumerate(ids) if t in (eos_id_val, eot_id_val)), None)
                return ep is not None and mask_id_val not in ids[:ep]
            return False

        if out is not None and not valid:
            gen_start_ac = prompt_ids.shape[1]

            # ── Pass 1 ───────────────────────────────────────────────────────
            if dp_consume_idx is not None:
                # DP: checker state is consistent with dp_consume_idx.
                consume_idx_ac = dp_consume_idx
            else:
                gen_ids = out[0, gen_start_ac:].tolist()
                first_mask = next(
                    (j for j, t in enumerate(gen_ids) if t == mask_id_val), len(gen_ids)
                )
                consume_idx_ac = gen_start_ac + first_mask

            _ac_deadline = min(time.monotonic() + 60.0, _instance_deadline)
            out, ac_steps, ac_mask_ms, ac_fwd_ms, ac_consume_idx, _ = autocomplete_greedy(
                model, out, checker, consume_idx_ac, gen_start_ac,
                mask_id=mask_id_val, eos_id=eos_id_val,
                max_steps=64, deadline=_ac_deadline,
            )
            valid = _recheck_valid(out, gen_start_ac)

            # ── Pass 2: iterative budget-exhaustion extension ─────────────────
            # Triggered when: still invalid AND grammar not yet accepting.
            # Extends by EXTENSION_CHUNK masked tokens and runs autocomplete
            # starting from ac_consume_idx — the checker's actual frontier
            # returned by the previous autocomplete call.  Using the checker
            # frontier (not the first mask in the tensor) keeps grammar state
            # in sync with token positions across multiple extension rounds.
            for _ext_round in range(MAX_EXTENSIONS):
                if valid or checker.is_accepting():
                    break
                if time.monotonic() > _ac_deadline:
                    break

                ext = torch.full(
                    (1, EXTENSION_CHUNK), mask_id_val,
                    dtype=out.dtype, device=out.device,
                )
                out_ext = torch.cat([out, ext], dim=1)
                out_ext, ac_steps2, ac_mask_ms2, ac_fwd_ms2, ac_consume_idx, _ = autocomplete_greedy(
                    model, out_ext, checker, ac_consume_idx, gen_start_ac,
                    mask_id=mask_id_val, eos_id=eos_id_val,
                    closing_bonus=100.0,
                    max_steps=64, deadline=_ac_deadline,
                )
                ac_steps   += ac_steps2
                ac_mask_ms += ac_mask_ms2
                ac_fwd_ms  += ac_fwd_ms2
                out    = out_ext
                valid  = _recheck_valid(out, gen_start_ac)

            # ── Pass 3: grammar-only force-close ─────────────────────────────
            # If still invalid after all extension rounds, try a grammar-only
            # DFS to find the minimal closing token sequence (no LM logits).
            # This handles deeply-nested JSON that needs e.g. `]}}` to close,
            # where the 1-step closing_bonus in Pass 2 never fires.
            if not valid and not checker.is_accepting() and time.monotonic() < _instance_deadline:
                actual_vocab_fc = 126464
                iid = getattr(instance, '_instance_id', None) or getattr(instance, 'instance_id', lambda: '?')()
                _fc_t0 = time.monotonic()
                # 5s deadline: easy instances finish in 1-3s (o61593=2.2s, o9886=1.2s);
                # hard instances (post-dp checker has expensive compute_mask ~100-500ms/call)
                # fail quickly → pass3b.  Was 15s before — wasted time on hard cases.
                _fc_deadline = _fc_t0 + 5.0
                closing_seq = _force_close_grammar(
                    checker, actual_vocab_fc,
                    max_steps=512,
                    priority_ids=_fc_priority_ids,
                    tokenizer=tokenizer,
                    deadline=_fc_deadline,
                )
                print(f"  [force_close] {time.monotonic()-_fc_t0:.1f}s  seq_len={len(closing_seq) if closing_seq else None}")
                if closing_seq:
                    # Append closing tokens + EOS so _recheck_valid sees a
                    # properly terminated sequence.
                    close_ids = closing_seq + [eos_id_val]
                    close_t = torch.tensor(
                        close_ids, dtype=out.dtype, device=out.device
                    ).unsqueeze(0)
                    out = torch.cat([out, close_t], dim=1)
                    # advance checker state through the closing tokens only
                    for tid in closing_seq:
                        checker.matcher.try_consume_tokens([tid])
                    valid = _recheck_valid(out, gen_start_ac)

                # ── Pass 3b: schema-based minimal JSON fallback ───────────────
                # If the grammar walk cycled (force_close returned None), generate
                # a minimal valid JSON directly from the schema string and replace
                # the corrupted generation.  This handles schemas where
                # additionalProperties:true prevents the grammar walk from knowing
                # which specific property names are required.
                if not valid:
                    try:
                        # Fresh checker from root state.  The generation region
                        # of prompt_ids contains only mask tokens (no real prefix
                        # to consume), so we just clone the schema's initial state.
                        def _p3b_fresh():
                            return cached_checker.clone()

                        # ── Fast path: grammar-guided encode of _minimal_json_value ──
                        schema_obj = json.loads(schema_str)
                        min_val = _minimal_json_value(schema_obj, root_schema=schema_obj)
                        min_str = json.dumps(min_val, ensure_ascii=False, separators=(',', ':'))
                        min_ids = _grammar_guided_encode(
                            _p3b_fresh(), min_str, tokenizer, vocab_size=126464,
                        )
                        if min_ids is not None:
                            vf = _p3b_fresh()
                            n_ok = vf.matcher.try_consume_tokens(min_ids)
                            if n_ok == len(min_ids) and vf.is_accepting():
                                min_close_t = torch.tensor(
                                    min_ids + [eos_id_val], dtype=out.dtype, device=out.device,
                                ).unsqueeze(0)
                                out = torch.cat([out[:, :gen_start_ac], min_close_t], dim=1)
                                valid = _recheck_valid(out, gen_start_ac)

                        # ── Medium path: grammar-walk from fresh root state ───────────
                        # When guided_encode fails (wrong property order or format),
                        # run _force_close_grammar from a FRESH checker.  The grammar
                        # itself enforces the correct ordering (n_valid=1 = forced
                        # sequence) and valid format chars (compute_mask restricts).
                        # This produces a complete valid JSON without model logits.
                        # Faster than autocomplete_greedy (no GPU forward passes).
                        if not valid and not min_ids:
                            _fc_fresh_t0 = time.monotonic()
                            _fc_fresh_chk = _p3b_fresh()
                            _fc_fresh_seq = _force_close_grammar(
                                _fc_fresh_chk, 126464,
                                max_steps=512,
                                priority_ids=_fc_priority_ids,
                                tokenizer=tokenizer,
                                deadline=_fc_fresh_t0 + 15.0,
                            )
                            _fc_fresh_secs = time.monotonic() - _fc_fresh_t0
                            print(f"  [pass3b_fc] seq_len={len(_fc_fresh_seq) if _fc_fresh_seq else None} "
                                  f"time={_fc_fresh_secs:.1f}s")
                            if _fc_fresh_seq:
                                _vf = _p3b_fresh()
                                _n_ok = _vf.matcher.try_consume_tokens(_fc_fresh_seq)
                                if _n_ok == len(_fc_fresh_seq) and _vf.is_accepting():
                                    _fc_close_t = torch.tensor(
                                        _fc_fresh_seq + [eos_id_val],
                                        dtype=out.dtype, device=out.device,
                                    ).unsqueeze(0)
                                    out = torch.cat([out[:, :gen_start_ac], _fc_close_t], dim=1)
                                    valid = _recheck_valid(out, gen_start_ac)

                        # ── Slow path: autocomplete from fresh checker state ──────────
                        if not valid:
                            _p3b_t0 = time.monotonic()
                            _p3b_chk = _p3b_fresh()
                            out_p3b = torch.full(
                                (1, gen_start_ac + 768), mask_id_val,
                                dtype=torch.long, device=model.device,
                            )
                            out_p3b[0, :gen_start_ac] = prompt_ids[0].to(model.device)
                            # closing_bonus=100 strongly prefers closing at comma/brace
                            # positions (len(valid_ids)<=32 guard keeps it fast).
                            out_p3b, _p3b_steps, _p3b_mask_ms, _p3b_fwd_ms, _p3b_cidx, _p3b_tc_ms = autocomplete_greedy(
                                model, out_p3b, _p3b_chk, gen_start_ac, gen_start_ac,
                                mask_id=mask_id_val, eos_id=eos_id_val,
                                closing_bonus=100.0, max_steps=512,
                                deadline=_p3b_t0 + 60.0,
                                closing_token_ids=_fc_closing_ids,
                            )
                            if _recheck_valid(out_p3b, gen_start_ac):
                                out = out_p3b
                                valid = True
                            _p3b_total_ms = (time.monotonic() - _p3b_t0) * 1000
                            print(f"  [pass3b] fast={'ok' if min_ids else 'fail'} ac_valid={valid} "
                                  f"steps={_p3b_steps} mask={_p3b_mask_ms:.0f}ms fwd={_p3b_fwd_ms:.0f}ms "
                                  f"tc={_p3b_tc_ms:.0f}ms total={_p3b_total_ms:.0f}ms")
                    except Exception as e:
                        print(f"  [pass3b] exception: {e}")

            elif not valid and checker.is_accepting():
                pass  # checker accepted but no EOS placed — already handled by pass 1/2

        elapsed = time.monotonic() - start_time

        if out is None:
            code = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]

        extracted = instance.extract_result(suffix_str + start_line + code)

        timing = STATS.summary()
        timing["autocomplete_steps"] = ac_steps
        timing["autocomplete_mask_ms"] = ac_mask_ms
        timing["autocomplete_fwd_ms"] = ac_fwd_ms
        # For async, effective constraint = gc + token_select + mask_wait + ac_mask
        # mask_compute was hidden behind forward pass
        effective_constraint_ms = (
            timing["grammar_check_total_ms"]
            + timing["token_select_total_ms"]
            + timing["mask_wait_total_ms"]
            + ac_mask_ms
        )
        total_constraint_ms = (
            timing["grammar_check_total_ms"]
            + timing["token_select_total_ms"]
            + timing["mask_compute_total_ms"]
        )
        total_forward_ms = timing["forward_total_ms"]

        result = {
            "instance_id": instance.instance_id(),
            "method": "dgrammar_dp" if method == "dp" else "dgrammar_v2_async",
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": len(resamples),
            "timing": {
                **timing,
                "total_constraint_ms": total_constraint_ms,
                "effective_constraint_ms": effective_constraint_ms,
                "total_forward_ms": total_forward_ms,
                "constraint_pct": (total_constraint_ms / (total_constraint_ms + total_forward_ms) * 100)
                    if (total_constraint_ms + total_forward_ms) > 0 else 0,
                "effective_constraint_pct": (effective_constraint_ms / (effective_constraint_ms + total_forward_ms) * 100)
                    if (effective_constraint_ms + total_forward_ms) > 0 else 0,
                "per_token_constraint_ms": (effective_constraint_ms / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
                "per_token_total_ms": (elapsed * 1000 / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
                "mask_time_saved_ms": timing["mask_compute_total_ms"] - timing["mask_wait_total_ms"],
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        torch.cuda.empty_cache()

        gc_mean = timing["grammar_check_mean_ms"]
        fwd_mean = timing["forward_mean_ms"]
        eff_pct = result["timing"]["effective_constraint_pct"]
        avg_batch = timing["avg_batch_size"]
        wait_mean = timing["mask_wait_mean_ms"]
        mc_mean = timing["mask_compute_mean_ms"]
        saved = result["timing"]["mask_time_saved_ms"]
        ac_info = f", AC={ac_steps}steps/{ac_mask_ms:.0f}ms" if ac_steps > 0 else ""
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, time={elapsed:.1f}s, "
            f"fwd={fwd_mean:.0f}ms(x{timing['forward_count']}), "
            f"gc={gc_mean:.3f}ms(x{timing['grammar_check_count']}), "
            f"mask={mc_mean:.2f}ms(x{timing['mask_compute_count']}), "
            f"wait={wait_mean:.2f}ms(x{timing['mask_wait_count']}), "
            f"overlap={timing['overlap_count']}, saved={saved:.0f}ms, "
            f"eff_constraint={eff_pct:.1f}%, batch={avg_batch:.1f}{ac_info}"
        )


if __name__ == "__main__":
    main()

"""
Bidirectional gap Viterbi on a DFA (DINGO-style).

A left anchor q_left + k mask positions + a fixed right suffix is a chain
dependency: forward Viterbi over the masks meets backward feasibility from the
suffix. The suffix is deterministic (no model probability), so the backward
``DP'' is: states q such that δ*(q, suffix) ∈ L (live).

``segmented_bidirectional_dingo`` handles alternating fixed tokens and mask
blocks (e.g. M1, c, d, M2) in one log-space Viterbi pass.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict

from csr_dfa import CSRTransitionMatrix
from sparse_dingo import DINGOResult, compute_transition_costs_sparse, sparse_dingo_dp

_CACHE_ATTR = "_bidi_suffix_compat_cache"


def clear_suffix_compat_cache(csr: CSRTransitionMatrix | None = None) -> None:
    """Drop cached suffix-compat sets for this CSR (no global registry)."""
    if csr is not None and hasattr(csr, _CACHE_ATTR):
        delattr(csr, _CACHE_ATTR)


def dfa_run(csr: CSRTransitionMatrix, start_state: int, tokens: list[int]) -> int | None:
    """Consume ``tokens`` from ``start_state``; return final state or None if stuck."""
    q = start_state
    for t in tokens:
        nxt: int | None = None
        for tok, q_next in csr.get_transitions(q):
            if tok == t:
                nxt = q_next
                break
        if nxt is None:
            return None
        q = nxt
    return q


def _dfa_step(csr: CSRTransitionMatrix, state: int, token: int) -> int | None:
    for tok, q_next in csr.get_transitions(state):
        if tok == token:
            return q_next
    return None


def states_compatible_with_suffix(
    csr: CSRTransitionMatrix,
    suffix_tokens: list[int],
    live_states: set[int],
) -> set[int]:
    """
    States q such that after consuming ``suffix_tokens`` from q, the DFA is in
    a live (accepting) state — i.e. valid meeting states before the suffix.
    """
    key = (tuple(suffix_tokens), frozenset(live_states))
    bucket: dict[tuple[tuple[int, ...], frozenset[int]], set[int]] = getattr(
        csr, _CACHE_ATTR, {}
    )
    if key in bucket:
        return bucket[key]

    good: set[int] = set()
    for q in range(csr.num_states):
        end = dfa_run(csr, q, suffix_tokens)
        if end is not None and end in live_states:
            good.add(q)
    bucket = dict(bucket)
    bucket[key] = good
    setattr(csr, _CACHE_ATTR, bucket)
    return good


def bidirectional_gap_dingo(
    csr: CSRTransitionMatrix,
    q_left: int,
    prob_vectors: list[list[float]],
    suffix_tokens: list[int],
    live_states: set[int],
) -> DINGOResult:
    """
    Max-product Viterbi on k mask positions from ``q_left``, then fixed suffix.

    ``final_state`` is the DFA state **after** consuming ``suffix_tokens`` (so
    callers can continue decoding).  ``probability`` is still the product over
    mask positions only (suffix has no model probability).
    """
    if not suffix_tokens:
        return sparse_dingo_dp(csr, prob_vectors, start_state=q_left, live_states=live_states)

    compat = states_compatible_with_suffix(csr, suffix_tokens, live_states)
    if not compat:
        return DINGOResult(tokens=[], final_state=q_left, probability=0.0, success=False)

    if not prob_vectors:
        end = dfa_run(csr, q_left, suffix_tokens)
        if end is None or end not in live_states:
            return DINGOResult(tokens=[], final_state=q_left, probability=0.0, success=False)
        return DINGOResult(tokens=[], final_state=end, probability=1.0, success=True)

    result = sparse_dingo_dp(csr, prob_vectors, start_state=q_left, live_states=compat)
    if not result.success:
        return result

    true_final = dfa_run(csr, result.final_state, suffix_tokens)
    ok = true_final is not None and true_final in live_states
    return DINGOResult(
        tokens=result.tokens,
        final_state=true_final if true_final is not None else result.final_state,
        probability=result.probability,
        success=ok,
    )


def _log_p(p: float) -> float:
    return math.log(p) if p > 0.0 else float("-inf")


class FixedSeg(TypedDict):
    type: Literal["fixed"]
    tokens: list[int]


class MaskSeg(TypedDict):
    type: Literal["mask"]
    probs: list[list[float]]


Segment = FixedSeg | MaskSeg


def segmented_bidirectional_dingo(
    csr: CSRTransitionMatrix,
    q_start: int,
    segments: list[Segment],
    suffix_tokens: list[int],
    live_states: set[int],
) -> DINGOResult:
    """
    Viterbi over alternating fixed-token runs and mask blocks, optional suffix.

    Each ``fixed`` segment advances the DFA deterministically (log mass unchanged).
    Each ``mask`` segment uses the same max-product token choice as
    ``sparse_dingo_dp`` for every mask column in ``probs``.
    """
    if not suffix_tokens:
        terminal = live_states
    else:
        terminal = states_compatible_with_suffix(csr, suffix_tokens, live_states)
        if not terminal:
            return DINGOResult(tokens=[], final_state=q_start, probability=0.0, success=False)

    flat: list[tuple[Literal["fixed", "mask"], Any]] = []
    for seg in segments:
        if seg["type"] == "fixed":
            for t in seg["tokens"]:
                flat.append(("fixed", t))
        else:
            for pv in seg["probs"]:
                flat.append(("mask", pv))

    if not flat:
        if not suffix_tokens:
            return DINGOResult(
                tokens=[], final_state=q_start, probability=1.0, success=q_start in live_states
            )
        end = dfa_run(csr, q_start, suffix_tokens)
        ok = end is not None and end in live_states
        return DINGOResult(
            tokens=[],
            final_state=end if end is not None else q_start,
            probability=1.0 if ok else 0.0,
            success=ok,
        )

    nq = csr.num_states
    neg_inf = float("-inf")
    w: list[float] = [neg_inf] * nq
    w[q_start] = 0.0
    pr: list[list[tuple[int | None, int | None]]] = []

    vocab_size = csr.vocab_size
    for kind, payload in flat:
        if kind == "fixed":
            t = int(payload)
            w_new = [neg_inf] * nq
            row: list[tuple[int | None, int | None]] = [(None, None)] * nq
            for q_old in range(nq):
                if w[q_old] == neg_inf:
                    continue
                nxt = _dfa_step(csr, q_old, t)
                if nxt is None:
                    continue
                if w[q_old] > w_new[nxt]:
                    w_new[nxt] = w[q_old]
                    row[nxt] = (q_old, t)
            w = w_new
            pr.append(row)
        else:
            pv: list[float] = payload
            Vi, Ti = compute_transition_costs_sparse(csr, pv, max(len(pv), vocab_size))
            w_new = [neg_inf] * nq
            row = [(None, None)] * nq
            for q in range(nq):
                best = neg_inf
                best_prev: tuple[int | None, int | None] = (None, None)
                for q_prime in range(nq):
                    key = (q, q_prime)
                    if key not in Vi:
                        continue
                    prev = w[q_prime]
                    if prev == neg_inf:
                        continue
                    cand = prev + _log_p(Vi[key])
                    if cand > best:
                        best = cand
                        best_prev = (q_prime, Ti[key])
                w_new[q] = best
                row[q] = best_prev
            w = w_new
            pr.append(row)

    max_log = neg_inf
    q_max = -1
    for q in terminal:
        if 0 <= q < nq and w[q] > max_log:
            max_log = w[q]
            q_max = q

    if q_max < 0 or not math.isfinite(max_log):
        return DINGOResult(tokens=[], final_state=q_start, probability=0.0, success=False)

    tokens: list[int] = []
    q_curr = q_max
    for row in reversed(pr):
        prev_q, tok = row[q_curr]
        if prev_q is None:
            continue
        if tok is not None:
            tokens.append(tok)
        q_curr = prev_q
    tokens.reverse()

    mask_final = q_max
    if suffix_tokens:
        true_final = dfa_run(csr, mask_final, suffix_tokens)
        ok = true_final is not None and true_final in live_states
    else:
        true_final = mask_final
        ok = mask_final in live_states

    return DINGOResult(
        tokens=tokens,
        final_state=true_final if true_final is not None else mask_final,
        probability=math.exp(max_log),
        success=ok,
    )

"""
Bidirectional gap Viterbi on a DFA (meet-in-the-middle with a fixed right suffix).

Fills k masked positions with tokens maximizing product of per-step model probabilities
subject to: δ*(q_left, gap_tokens) = q and δ*(q, suffix) ∈ live_states.

This replaces LAVE-style random lookahead with a deterministic DP when a fixed
suffix to the right is known (diffusion partially unmasked sequence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sparse_dingo import compute_transition_costs_sparse

if TYPE_CHECKING:
    from csr_dfa import CSRTransitionMatrix


@dataclass
class BidirectionalResult:
    tokens: list[int]
    final_state: int
    probability: float
    success: bool


def dfa_step(csr: "CSRTransitionMatrix", q: int, t: int) -> int | None:
    for tok, qn in csr.get_transitions(q):
        if tok == t:
            return qn
    return None


def dfa_run(csr: "CSRTransitionMatrix", q0: int, token_ids: list[int]) -> int | None:
    q = q0
    for t in token_ids:
        qn = dfa_step(csr, q, t)
        if qn is None:
            return None
        q = qn
    return q


def states_compatible_with_suffix(
    csr: "CSRTransitionMatrix",
    suffix_tokens: list[int],
    live_states: set[int],
) -> set[int]:
    """States q such that consuming suffix_tokens from q ends in live_states."""
    if not suffix_tokens:
        return set(live_states)
    good: set[int] = set()
    for q in range(csr.num_states):
        qf = dfa_run(csr, q, suffix_tokens)
        if qf is not None and qf in live_states:
            good.add(q)
    return good


def bidirectional_gap_dingo(
    csr: "CSRTransitionMatrix",
    q_left: int,
    prob_vectors: list[list[float]],
    suffix_tokens: list[int],
    live_states: set[int],
) -> BidirectionalResult:
    """
    k-step forward Viterbi from q_left; keep only paths ending in a state that
    can still consume suffix_tokens into an accepting/live state.
    """
    k = len(prob_vectors)
    if k == 0:
        qf = dfa_run(csr, q_left, suffix_tokens)
        ok = qf is not None and qf in live_states
        return BidirectionalResult(
            tokens=[], final_state=q_left, probability=1.0, success=ok
        )

    feasible_end = states_compatible_with_suffix(csr, suffix_tokens, live_states)
    if not feasible_end:
        return BidirectionalResult(
            tokens=[], final_state=q_left, probability=0.0, success=False
        )

    vocab_size = len(prob_vectors[0]) if prob_vectors else csr.vocab_size
    W: dict[tuple[int, int], float] = {}
    Pr: dict[tuple[int, int], tuple[int | None, int | None]] = {}

    for q in range(csr.num_states):
        W[(0, q)] = 0.0
        Pr[(0, q)] = (None, None)
    W[(0, q_left)] = 1.0

    all_Vi: list = []
    all_Ti: list = []
    for i in range(k):
        Vi, Ti = compute_transition_costs_sparse(csr, prob_vectors[i], vocab_size)
        all_Vi.append(Vi)
        all_Ti.append(Ti)

    for i in range(1, k + 1):
        Vi = all_Vi[i - 1]
        Ti = all_Ti[i - 1]
        for q in range(csr.num_states):
            best_val = 0.0
            best_prev: tuple[int | None, int | None] = (None, None)
            for q_prime in range(csr.num_states):
                key = (q, q_prime)
                if key not in Vi:
                    continue
                prev_prob = W.get((i - 1, q_prime), 0.0)
                cand = prev_prob * Vi[key]
                if cand > best_val:
                    best_val = cand
                    best_prev = (q_prime, Ti[key])
            W[(i, q)] = best_val
            Pr[(i, q)] = best_prev

    q_max = -1
    max_prob = 0.0
    for q in feasible_end:
        p = W.get((k, q), 0.0)
        if p > max_prob:
            max_prob = p
            q_max = q

    if q_max < 0 or max_prob <= 0:
        return BidirectionalResult(
            tokens=[], final_state=q_left, probability=0.0, success=False
        )

    tokens: list[int] = []
    q_curr = q_max
    for i in range(k, 0, -1):
        q_prev, t = Pr[(i, q_curr)]
        if t is not None:
            tokens.append(t)
        q_curr = q_prev if q_prev is not None else q_curr
    tokens.reverse()
    return BidirectionalResult(
        tokens=tokens, final_state=q_max, probability=max_prob, success=True
    )

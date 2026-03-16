"""
B1: Original DINGO - O(N) Baseline

Implements the naive DINGO algorithm that scans the full vocabulary V (size N)
for each transition cost computation. Used as the baseline for benchmarking
against B2 (STATIC + DINGO) which achieves O(K) via sparse indexing.

Reference: DINGO (Suresh et al., NeurIPS 2025) - Algorithm 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class DINGOResult:
    """Result of DINGO constrained decoding."""
    tokens: list[int]
    final_state: int
    probability: float
    success: bool


def compute_transition_costs_naive(
    num_states: int,
    vocab_size: int,
    transition_fn: Callable[[int, int], int | None],
    prob_vector: list[float],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """
    Compute V_i(q, q') and T_i(q, q') using O(|Q| * N) - full vocabulary scan.
    
    V_i(q, q') = max_{t∈V} v_i(t) s.t. q ∈ δ(q', t)
    
    For each (q', q), we iterate over ALL N tokens in the vocabulary.
    This is the bottleneck that STATIC's CSR format eliminates.
    
    Complexity: O(|Q| * N)
    """
    Vi: dict[tuple[int, int], float] = {}
    Ti: dict[tuple[int, int], int] = {}
    
    for q_prime in range(num_states):
        for q in range(num_states):
            best_prob = 0.0
            best_tok = -1
            
            # O(N): Scan entire vocabulary
            for t in range(vocab_size):
                q_next = transition_fn(q_prime, t)
                if q_next == q and q_next is not None:
                    prob = prob_vector[t] if t < len(prob_vector) else 0.0
                    if prob > best_prob:
                        best_prob = prob
                        best_tok = t
            
            if best_tok >= 0:
                Vi[(q, q_prime)] = best_prob
                Ti[(q, q_prime)] = best_tok
    
    return Vi, Ti


def baseline_dingo_dp(
    num_states: int,
    vocab_size: int,
    transition_fn: Callable[[int, int], int | None],
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    block_length: int | None = None,
) -> DINGOResult:
    """
    Original DINGO DP with O(N) transition cost computation.
    
    Same DP logic as sparse_dingo_dp, but uses full vocabulary scan
    for V_i(q, q') instead of CSR sparse lookup.
    """
    d = block_length if block_length is not None else len(prob_vectors)
    if d == 0:
        return DINGOResult(tokens=[], final_state=start_state, probability=1.0, success=True)
    
    # Initialize DP tables
    W: dict[tuple[int, int], float] = {}
    Pr: dict[tuple[int, int], tuple[int | None, int | None]] = {}
    
    for q in range(num_states):
        W[(0, q)] = 0.0
        Pr[(0, q)] = (None, None)
    W[(0, start_state)] = 1.0
    
    # Precompute transition costs - O(d * |Q| * N) total
    all_Vi: list[dict[tuple[int, int], float]] = []
    all_Ti: list[dict[tuple[int, int], int]] = []
    
    for i in range(d):
        Vi, Ti = compute_transition_costs_naive(
            num_states, vocab_size, transition_fn, prob_vectors[i]
        )
        all_Vi.append(Vi)
        all_Ti.append(Ti)
    
    # DP forward pass (same as sparse)
    for i in range(1, d + 1):
        Vi = all_Vi[i - 1]
        Ti = all_Ti[i - 1]
        
        for q in range(num_states):
            best_val = 0.0
            best_prev: tuple[int | None, int | None] = (None, None)
            
            for q_prime in range(num_states):
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
    
    # Find best live state
    q_max = -1
    max_prob = 0.0
    for q in live_states:
        p = W.get((d, q), 0.0)
        if p > max_prob:
            max_prob = p
            q_max = q
    
    if q_max < 0 or max_prob <= 0:
        return DINGOResult(
            tokens=[],
            final_state=start_state,
            probability=0.0,
            success=False,
        )
    
    # Backtrack
    tokens: list[int] = []
    q_curr = q_max
    for i in range(d, 0, -1):
        q_prev, t = Pr[(i, q_curr)]
        if t is not None:
            tokens.append(t)
        q_curr = q_prev if q_prev is not None else q_curr
    
    tokens.reverse()
    return DINGOResult(
        tokens=tokens,
        final_state=q_max,
        probability=max_prob,
        success=True,
    )

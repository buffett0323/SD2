"""
Sparse DINGO: O(K) Dynamic Programming for Constrained Decoding.

Optimizes DINGO's transition cost computation from O(N) to O(K) by using
STATIC's CSR sparse indexing. Instead of scanning the full vocabulary V (size N)
for V_i(q, q') = max_{t∈V} v_i(t) s.t. q ∈ δ(q', t), we only iterate over the
K valid tokens stored in the CSR slice for state q'.

Reference: 
- DINGO (Suresh et al., NeurIPS 2025) - Constrained Inference for Diffusion LLMs
- STATIC (Su et al., 2026) - Sparse Transition Matrix for Constrained Decoding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .csr_dfa import CSRTransitionMatrix


@dataclass 
class DINGOResult:
    """Result of DINGO constrained decoding."""
    tokens: list[int]
    final_state: int
    probability: float
    success: bool


def compute_transition_costs_sparse(
    csr: CSRTransitionMatrix,
    prob_vector: list[float],
    vocab_size: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """
    Compute V_i(q, q') and T_i(q, q') using O(|Q| * K) instead of O(|Q| * N).
    
    V_i(q, q') = max_{t∈V} v_i(t) s.t. q ∈ δ(q', t)
    T_i(q, q') = argmax token for the above
    
    Instead of iterating over all N tokens for each (q', q), we use the CSR slice:
    For source state q', only K tokens have valid transitions. We iterate only over those.
    
    Complexity: O(|Q| * K) vs original O(|Q| * N)
    """
    Vi: dict[tuple[int, int], float] = {}
    Ti: dict[tuple[int, int], int] = {}
    
    for q_prime in range(csr.num_states):
        # O(K): Get only valid (token, next_state) pairs for q'
        transitions = csr.get_transitions(q_prime)
        
        # Group by target state q, keep max probability token
        best_for_target: dict[int, tuple[float, int]] = {}
        
        for t, q_next in transitions:
            if t < len(prob_vector):
                prob = prob_vector[t]
            else:
                prob = 0.0
            
            if q_next not in best_for_target or prob > best_for_target[q_next][0]:
                best_for_target[q_next] = (prob, t)
        
        for q, (max_prob, best_tok) in best_for_target.items():
            Vi[(q, q_prime)] = max_prob
            Ti[(q, q_prime)] = best_tok
    
    return Vi, Ti


def sparse_dingo_dp(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    block_length: int | None = None,
) -> DINGOResult:
    """
    DINGO Dynamic Programming with O(K) sparse transition cost computation.
    
    Args:
        csr: CSR format DFA (STATIC sparse indexing)
        prob_vectors: v_1, ..., v_d - probability vectors at each position
        start_state: q0
        live_states: Ql - states that can reach accepting states
        block_length: d (default: len(prob_vectors))
    
    Returns:
        DINGOResult with optimal token sequence
    """
    d = block_length if block_length is not None else len(prob_vectors)
    if d == 0:
        return DINGOResult(tokens=[], final_state=start_state, probability=1.0, success=True)
    
    vocab_size = len(prob_vectors[0]) if prob_vectors else csr.vocab_size
    
    # Initialize DP tables
    # W[i, q] = max probability to reach state q in i steps
    W: dict[tuple[int, int], float] = {}
    Pr: dict[tuple[int, int], tuple[int | None, int | None]] = {}
    
    for q in range(csr.num_states):
        W[(0, q)] = 0.0
        Pr[(0, q)] = (None, None)
    W[(0, start_state)] = 1.0
    
    # Precompute transition costs for each position (O(d * |Q| * K) total)
    all_Vi: list[dict[tuple[int, int], float]] = []
    all_Ti: list[dict[tuple[int, int], int]] = []
    
    for i in range(d):
        Vi, Ti = compute_transition_costs_sparse(
            csr, prob_vectors[i], vocab_size
        )
        all_Vi.append(Vi)
        all_Ti.append(Ti)
    
    # DP forward pass
    for i in range(1, d + 1):
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
    
    # Backtrack to get optimal sequence
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

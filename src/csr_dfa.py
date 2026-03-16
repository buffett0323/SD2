"""
CSR (Compressed Sparse Row) representation of DFA for STATIC-style sparse indexing.

Converts a DFA transition function δ: Q × V → Q into a sparse format where
for each state q, we store only the K valid (token, next_state) pairs instead
of scanning the full vocabulary of size N.

Reference: STATIC (Su et al., 2026) - "Vectorizing the Trie: Efficient Constrained
Decoding for LLM-based Generative Retrieval on Accelerators"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CSRTransitionMatrix:
    """
    Compressed Sparse Row format for DFA transitions.
    
    For each state q ∈ Q:
    - row_pointers[q] to row_pointers[q+1] defines a slice
    - column_indices[slice] = token IDs that enable valid transitions from q
    - values[slice] = target states reached by those tokens
    
    This reduces lookup from O(N) to O(K) where K = number of valid transitions from q.
    """
    row_pointers: list[int]   # P: start index for each state, length |Q|+1
    column_indices: list[int]  # C: token IDs, length = total transitions
    values: list[int]          # V: target state IDs, length = total transitions
    num_states: int
    vocab_size: int
    
    def get_valid_tokens(self, state: int) -> list[int]:
        """Get the K valid token indices for state q. O(K) instead of O(N)."""
        start = self.row_pointers[state]
        end = self.row_pointers[state + 1]
        return self.column_indices[start:end]
    
    def get_transitions(self, state: int) -> list[tuple[int, int]]:
        """Get (token_id, next_state) pairs for state q. O(K) lookup."""
        start = self.row_pointers[state]
        end = self.row_pointers[state + 1]
        return list(zip(
            self.column_indices[start:end],
            self.values[start:end]
        ))
    
    def get_num_valid_tokens(self, state: int) -> int:
        """K = number of valid tokens for state q."""
        return self.row_pointers[state + 1] - self.row_pointers[state]
    
    def max_branch_factor(self) -> int:
        """Max K across all states (for fixed-size slicing in vectorized ops)."""
        return max(
            self.get_num_valid_tokens(q) 
            for q in range(self.num_states)
        ) if self.num_states > 0 else 0


def build_csr_from_dfa(
    num_states: int,
    vocab_size: int,
    transition_fn: Callable[[int, int], int | None],
) -> CSRTransitionMatrix:
    """
    Build CSR matrix from DFA transition function.
    
    Offline phase: Convert DFA δ: Q × V → Q into CSR format.
    Only stores (t, q') where δ(q, t) = q' is defined (non-sink).
    
    Args:
        num_states: |Q|
        vocab_size: |V| = N
        transition_fn: (state, token) -> next_state or None (invalid)
    
    Returns:
        CSRTransitionMatrix with O(K) lookup per state
    """
    row_pointers = [0]
    column_indices = []
    values = []
    
    for q in range(num_states):
        valid_pairs: list[tuple[int, int]] = []
        
        for t in range(vocab_size):
            q_next = transition_fn(q, t)
            if q_next is not None and q_next >= 0:
                valid_pairs.append((t, q_next))
        
        for t, q_next in valid_pairs:
            column_indices.append(t)
            values.append(q_next)
        
        row_pointers.append(len(column_indices))
    
    return CSRTransitionMatrix(
        row_pointers=row_pointers,
        column_indices=column_indices,
        values=values,
        num_states=num_states,
        vocab_size=vocab_size,
    )


def build_csr_from_transition_dict(
    transitions: dict[tuple[int, int], int],
    num_states: int,
    vocab_size: int,
) -> CSRTransitionMatrix:
    """
    Build CSR from explicit (state, token) -> next_state mapping.
    Useful for testing with small DFAs.
    """
    def transition_fn(q: int, t: int) -> int | None:
        return transitions.get((q, t))
    
    return build_csr_from_dfa(num_states, vocab_size, transition_fn)

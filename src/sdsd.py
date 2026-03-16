"""
Sparse Deterministic Speculative Decoding (SDSD)

Combines DINGO's dynamic programming logic with STATIC's sparse indexing to achieve
O(K) complexity for transition cost computation, where K << N (vocabulary size).

Key optimization:
- DINGO: V_i(q, q') = max_{t∈V} v_i(t) requires O(N) scan per (q', q)
- SDSD: Use CSR format - only iterate over K valid tokens per state
- Complexity: O(|Q| · K) instead of O(|Q| · N)

References:
- STATIC: Sparse Transition Matrix-Accelerated Trie Index (Su et al., 2026)
- DINGO: Constrained Inference for Diffusion LLMs (Suresh et al., NeurIPS 2025)
"""

from __future__ import annotations

from .csr_dfa import CSRTransitionMatrix, build_csr_from_dfa, build_csr_from_transition_dict
from .sparse_dingo import sparse_dingo_dp, DINGOResult, compute_transition_costs_sparse

__all__ = [
    "CSRTransitionMatrix",
    "build_csr_from_dfa",
    "build_csr_from_transition_dict",
    "sparse_dingo_dp",
    "DINGOResult",
    "compute_transition_costs_sparse",
]

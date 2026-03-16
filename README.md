# SDSD: Sparse Deterministic Speculative Decoding

Combines **DINGO**'s dynamic programming logic with **STATIC**'s sparse indexing to achieve **O(K)** complexity for constrained decoding, where K << N (vocabulary size).

## Key Optimization

| Component | Original (DINGO) | Optimized (SDSD) |
|-----------|------------------|------------------|
| Transition cost | \( V_i(q, q') = \max_{t \in V} v_i(t) \) | Same formula, but iterate only over K valid tokens |
| Complexity | O(\|Q\| · N) | O(\|Q\| · K) |
| I/O | Full vocabulary scan | CSR slice: `P[q]` to `P[q+1]` |

## Structure

```
src/
├── csr_dfa.py         # STATIC-style CSR format for DFA
├── sparse_dingo.py    # B2: O(K) DINGO DP (STATIC + DINGO)
├── baseline_dingo.py  # B1: O(N) DINGO baseline
├── benchmark_b1_b2.py # B1 vs B2 benchmark
├── test_dllm_sdsd.py  # DLLM integration (LLaDA/Dream)
├── sdsd.py            # Main module
└── test_sdsd.py       # Test suite
```

## Usage

```python
from sdsd import (
    build_csr_from_transition_dict,
    sparse_dingo_dp,
)

# Define DFA: (state, token) -> next_state
transitions = {(0, 0): 1, (0, 1): 0, (1, 2): 2}
csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=100)

# Probability vectors from model (e.g., diffusion LLM)
prob_vectors = [[0.9, 0.1, 0.0, ...], [0.0, 0.0, 1.0, ...]]

result = sparse_dingo_dp(
    csr, prob_vectors,
    start_state=0,
    live_states={2},
)
print(result.tokens)   # Optimal constrained sequence
print(result.probability)
```

## UV install
```bash
uv pip install -r requirements.txt 
uv sync
uv sync --extra dream # Dream-7B dependencies
```

## Run Tests

```bash
cd src && python test_sdsd.py
```

## Run B1 vs B2 Benchmark

```bash
cd src && python benchmark_b1_b2.py
```

## Run DLLM Integration Test

```bash
cd src && python test_dllm_sdsd.py --mock          # No GPU: synthetic logits
cd src && python test_dllm_sdsd.py --model dream  # Dream-7B (needs ~20GB GPU)
cd src && python test_dllm_sdsd.py --model llada  # LLaDA-8B-Instruct (needs ~16GB GPU)
```

Requires: `pip install torch transformers`. Dream needs transformers>=4.46; LLaDA needs transformers==4.38.2.

Output is written to `benchmark_results.txt`. Metrics:
- **Latency (ms)**: Total time per block decode
- **Speedup**: B2 vs B1 (expect 2–100x for N >> K)
- **Vocab scans**: N (B1) vs K (B2) per transition

## References

- **STATIC** (Su et al., 2026): Sparse Transition Matrix-Accelerated Trie Index
- **DINGO** (Suresh et al., NeurIPS 2025): Constrained Inference for Diffusion LLMs

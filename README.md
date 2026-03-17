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
├── herding.py         # B3: Herding momentum for blocked intentions
├── speculative_tree.py  # Self-Speculative Tree: NFE reduction
├── sdsd.py            # SDSD: full pipeline
├── benchmark_b1_b2.py # B1 vs B2 benchmark
├── test_dllm_sdsd.py  # DLLM integration (LLaDA/Dream)
├── test_sdsd.py       # Test suite
└── eval/
    ├── evaluate.py    # B1/B2/B3/SDSD evaluation
    ├── intent_recovery.py  # Intent recovery ablation
    └── metrics.py     # Parse rate, constraint satisfaction
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
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv pip install -r requirements.txt 
uv sync

# or Dream-7B dependencies
uv sync --extra dream
```

## Cloning constrained-diffusion
```bash
git clone https://github.com/eth-sri/constrained-diffusion.git
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
cd src && python test_dllm_sdsd.py --mock         # No GPU: synthetic logits
cd src && python test_dllm_sdsd.py --model dream  # Dream-7B (needs ~20GB GPU)
cd src && python test_dllm_sdsd.py --model llada  # LLaDA-8B-Instruct (needs ~16GB GPU)
```

Requires: `pip install torch transformers`. Dream needs transformers>=4.46; LLaDA needs transformers==4.38.2.

## Run B1/B2/B3/SDSD Evaluation

```bash
python run_experiments.py --model dream --mock         # Synthetic (no GPU)
python run_experiments.py --model dream --samples 10   # Dream-7B (GPU)
python run_experiments.py --model llada                # LLaDA-8B (GPU)
python run_experiments.py --intent-recovery-only       # Intent recovery ablation only
```

## Run Ablation Study

See [docs/ABLATION_EXPERIMENT_DESIGN.md](docs/ABLATION_EXPERIMENT_DESIGN.md) for full design.

```bash
python run_ablation.py --model dream --mock --samples 20   # Synthetic
python run_ablation.py --model dream --samples 20           # Dream-7B
python run_ablation.py --model dream --dataset json-mode-eval  # With dataset
```

Dataset download: [docs/DATASET_DOWNLOAD_GUIDE.md](docs/DATASET_DOWNLOAD_GUIDE.md)

Output: `results/experiment_results.json` (or `--output` path). Metrics: Latency, NFE, TTFT, Success rate, Intent recovery steps.

## Run Intent Recovery Ablation

Compares B2 (STATIC+DINGO) vs Herding (SDSD) on recovery after perturbing: inject low-prob token at step t, measure steps until high-prob token is chosen again. Herding recovers faster due to momentum preservation.

```bash
cd src && python -m eval.intent_recovery
```

## Benchmark Output

Output is written to `benchmark_results.txt`. Metrics:
- **Latency (ms)**: Total time per block decode
- **Speedup**: B2 vs B1 (expect 2–100x for N >> K)
- **Vocab scans**: N (B1) vs K (B2) per transition

## References

- **STATIC** (Su et al., 2026): Sparse Transition Matrix-Accelerated Trie Index
- **DINGO** (Suresh et al., NeurIPS 2025): Constrained Inference for Diffusion LLMs

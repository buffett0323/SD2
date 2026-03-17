# SDSD: Sparse Deterministic Speculative Decoding in Diffusion Language Models

SDSD combines **DINGO** (constrained decoding), **STATIC** (O(K) sparse indexing), **Herding** (intent recovery), and **Speculative Tree** (NFE reduction) for efficient grammar-constrained generation from diffusion LLMs.

## Key Components

| Component | Role | Effect |
|-----------|------|--------|
| **STATIC** | CSR sparse DFA | O(N) → O(K) transition lookup |
| **Herding** | Momentum for blocked intent | Faster intent recovery |
| **Speculative Tree** | Draft + verify in rounds | NFE = T/τ instead of T |

## Project Structure

```
src/
├── csr_dfa.py           # STATIC-style CSR format for DFA
├── baseline_dingo.py    # B1: O(N) DINGO baseline
├── sparse_dingo.py      # B2: O(K) DINGO (STATIC + DINGO)
├── herding.py           # B3: Herding momentum decoding
├── speculative_tree.py  # Speculative tree + sdsd_multi_round
├── sdsd.py              # SDSD pipeline exports
├── dataset_loaders.py   # JSON-Mode-Eval, HumanEval, MBPP, GSM-Symbolic
├── test_dllm_sdsd.py    # LLaDA/Dream integration
├── benchmark_b1_b2.py   # B1 vs B2 benchmark
├── test_sdsd.py         # Unit tests
└── eval/
    ├── evaluate.py
    ├── intent_recovery.py
    └── metrics.py
```

## Installation

```bash
# uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv pip install -r requirements.txt

# Dream-7B: transformers>=4.46
# LLaDA-8B: transformers==4.38.2 (see requirements-llada.txt)
```

## Run Ablation Study

The main experiment runner. Uses **JSON-Mode-Eval** by default, generates 64 tokens, and compares 5 methods: Baseline, STATIC+DINGO, Herding, Ablation3 (Spec-Tree), SDSD.

```bash
# Mock (no GPU, synthetic logits)
python run_ablation.py --model dream --mock --dataset json-mode-eval --samples 10

# Dream-7B (needs ~20GB GPU)
python run_ablation.py --model dream --dataset json-mode-eval --samples 20

# LLaDA-8B (needs ~16GB GPU)
python run_ablation.py --model llada --dataset json-mode-eval --samples 20

# With intent recovery experiment
python run_ablation.py --model dream --dataset json-mode-eval --intent-recovery

# Quick mock (16 tokens, smaller vocab)
python run_ablation.py --model dream --mock --quick --samples 5

# Custom output
python run_ablation.py --model dream --output results/ablation_results_dream.json
```

**Output**: `results/ablation_results.json` (or `--output` path). Metrics: TTFT, Throughput, NFE, Parse Rate, Intent Recovery steps.

See [docs/ablation.md](docs/ablation.md) for full design and results (Dream-7B, LLaDA-8B).

## Dataset

JSON-Mode-Eval is loaded automatically from Hugging Face. See [docs/download_data.md](docs/download_data.md) for other datasets.

```bash
# Datasets are fetched on first run; install: pip install datasets
```

## Run Tests

```bash
cd src && python test_sdsd.py
```

## Run DLLM Integration Test

```bash
cd src && python test_dllm_sdsd.py --mock         # No GPU
cd src && python test_dllm_sdsd.py --model dream  # Dream-7B (~20GB)
cd src && python test_dllm_sdsd.py --model llada  # LLaDA-8B (~16GB)
```

## Run B1 vs B2 Benchmark

```bash
cd src && python benchmark_b1_b2.py
```

Output: `benchmark_results.txt` (latency, speedup, vocab scans).

## Intent Recovery (Standalone)

```bash
cd src && python -m eval.intent_recovery
```

Compares B2 vs Herding on recovery after perturbing a token. Herding recovers faster via momentum.

## References

- **STATIC** (Su et al., 2026): Sparse Transition Matrix-Accelerated Trie Index
- **DINGO** (Suresh et al., NeurIPS 2025): Constrained Inference for Diffusion LLMs

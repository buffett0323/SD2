# SDSD: Sparse Deterministic Speculative Decoding in Diffusion Language Models

SDSD combines **DINGO** (constrained decoding), **STATIC** (O(K) sparse indexing), **Herding** (intent recovery), and **Speculative Tree** (NFE reduction) for efficient grammar-constrained generation from diffusion LLMs.

## TODO

- Add accuracy metrics (e.g., parse rate, constraint satisfaction, pass@1)
- Compare SDSD with LAVE (or other baselines)
- Benchmark latency with warmup: exclude first 10 runs to avoid cold-start overhead
- Implementation check for herding, STATIC

## Key Components


| Component            | Role                        | Effect                        |
| -------------------- | --------------------------- | ----------------------------- |
| **STATIC**           | CSR sparse DFA              | O(N) → O(K) transition lookup |
| **Herding**          | Momentum for blocked intent | Faster intent recovery        |
| **Speculative Tree** | Draft + verify in rounds    | NFE = T/τ instead of T        |


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

# Exclude first 10 runs from latency metrics (cold-start warmup)
python run_ablation.py --model dream --samples 30 --warmup 10

# Full json-mode-eval test (syntactic_accuracy, constraint_pct) — needs GPU
python run_ablation.py --model dream --dataset json-mode-eval --samples 30 --warmup 10
```

**Output**: `results/ablation_results.json` (or `--output` path). Metrics: TTFT, Throughput, NFE, Parse Rate, Syntactic Accuracy, Constraint %, Intent Recovery steps.

**Can syntactic_accuracy and constraint_pct be < 100%?**

- **Constraint %**: Always 100% for constrained methods — every emitted token is a valid DFA transition by construction.
- **Syntactic accuracy**: Can be < 100%. The DFA restricts *which* tokens are allowed, not *how* they combine. The model can produce valid-token sequences that decode to invalid JSON (e.g., `}{"`, truncated `{"a":`). With a strict JSON grammar DFA, you’d expect higher syntactic accuracy.

See [docs/ablation.md](docs/ablation.md) for full design and results (Dream-7B, LLaDA-8B).

## Unified Benchmark (SDSD vs Dgrammar vs LAVE)

Compare Baseline, Ablations, SDSD, Dgrammar, and LAVE on JSON-Bench (jsonschema) with Dgrammar-style metrics:

| Method | Syntactic | Functional | Mean Time | Median | P95 | Max | Constraint % |
|--------|-----------|------------|-----------|--------|-----|-----|--------------|

```bash
# Run SDSD methods only (Baseline, Ablation1, Ablation2, Ablation3, SDSD)
python run_unified_benchmark.py --methods baseline,ablation1,ablation2,ablation3,sdsd

# BiDi: bidirectional gap Viterbi on JSON DFA + fixed right suffix (vs LAVE-style sampling)
python run_unified_benchmark.py --methods bidi --limit 20 --output results/unified

# Quick test (20 instances)
python run_unified_benchmark.py --limit 20 --output results/unified

# Aggregate results (SDSD + Dgrammar + LAVE if available)
python aggregate_unified_results.py results/unified vendor/dgrammar/results

# Or use the shell script (runs SDSD, Dgrammar, LAVE, then aggregates)
bash run_unified_benchmark.sh

# SDSD/BiDi vs LAVE — same LLaDA + JSON-Bench + diffusion steps (documented knobs)
python run_lave_sdsd_compare.py --methods sdsd,bidi --limit 20 --output results/lave_sdsd_compare
python run_lave_sdsd_compare.py --methods sdsd,bidi --limit 272 --run-lave --aggregate   # needs vendor/dgrammar
```

See [docs/lave_sdsd_compare.md](docs/lave_sdsd_compare.md) for aligned config and prerequisites.

**Output**: `results/unified/unified_comparison.json` and printed table. Requires `vendor/CD4dLLM` for ETH syntactic/functional evaluation. See [docs/experiment_comparison.md](docs/experiment_comparison.md) for metric definitions.

## Dataset

JSON-Mode-Eval is loaded automatically from Hugging Face. See [docs/download_data.md](docs/download_data.md) for other datasets.



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

## Debugging Unified Benchmark

If SDSD produces garbage output while Dgrammar produces valid JSON:

1. **Run argmax (Dgrammar-style) to isolate the issue:**
   ```bash
   python run_unified_benchmark.py --methods argmax --limit 5
   ```
   If argmax also produces garbage → prompt format or dataset mismatch.
   If argmax produces valid JSON → bug is in our DINGO/Herding picker.

2. **ETH checker (Syntactic/Functional):** Requires `vendor/CD4dLLM`. Aggregate merges instance data (schema, input, output) for validation. If checker fails to load, Syntactic/Functional will be 0.

3. **Debug script:** `python debug_diffusion.py` compares Dgrammar vs SDSD on one instance (needs GPU + llguidance).

4. **ablation2 (Herding) in diffusion:** Uses a single `HerdingMomentumState` per instance so `w` persists across all frontier picks and violator retries within `generate_diffusion_sdsd` (not reset each step).

## Cloning Baseline Repos

```bash
# STATIC
git clone https://github.com/youtube/static-constraint-decoding.git

# Constrained diffusion
git clone https://github.com/eth-sri/constrained-diffusion.git

# LAVE
git clone https://github.com/zhangyitonggg/CD4dLLM.git

# DGrammar
git clone https://github.com/guan404ming/dgrammar.git
```

## References

- **STATIC** (Su et al., 2026): Sparse Transition Matrix-Accelerated Trie Index
- **DINGO** (Suresh et al., NeurIPS 2025): Constrained Inference for Diffusion LLMs
- **LAVE** (Zhang et al., ACM 2026): Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars
- **Constrained Decoding** (Mündler et al., NeurIPS 2025 DL4C Oral): Constrained Decoding of Diffusion LLMs with Context-Free Grammars


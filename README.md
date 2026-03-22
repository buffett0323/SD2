# Grammar-Guided Beam Search (GGBS) for constrained diffusion

This repo implements **GGBS** — grammar-guided beam search on top of the Dgrammar / LAVE-style constrained diffusion stack — plus optional **SDSD** (sparse DINGO + herding + speculative tree) baselines for comparison.

**Focus:** JSON-schema constrained generation with **LLaDA-8B**, diffusion **T = 128**, same dataset and bench setup as `vendor/dgrammar`.

## What GGBS does

GGBS replaces per-step sampling with a **segmented beam** over valid token prefixes: it uses the same `TokenChecker` / mask as LAVE, but organizes search as beam expansion with caching and optional suffix pruning. See `src/ggbs.py` and `vendor/dgrammar/bench/run_lave_ggbs.py`.

## JSON-Bench timed results (jsonschema, seed 0, T 128)

Aggregated from saved bench outputs (wall-clock `time_taken`, schema validity bit `valid`, and `timing.constraint_pct` as reported by each method):


| Method   | Syntactic | Functional | Mean Time (s) | Median (s) | P95 (s) | Max (s) | Constraint % |
| -------- | --------- | ---------- | ------------- | ---------- | ------- | ------- | ------------ |
| **LAVE** | 98.1%     | —          | 26.1          | 13.9       | 68.6    | 663.3   | 37.2         |
| **GGBS** | 98.4%     | —          | 31.7          | 14.1       | 82.7    | 1728.0  | 13.6         |


**Notes**

- **Sources:** `vendor/dgrammar/results/lave_timed_jsonschema_s0_t128.jsonl` (n = 213), `vendor/dgrammar/results/ggbs_lave_timed_jsonschema_s0_t128.jsonl` (n = 251). Subset and ordering follow that bench run; instance sets are not identical (211 overlapping `instance_id`s).
- **Syntactic:** fraction of rows with `valid: true` (schema check as produced by the bench).
- **Functional:** not stored in these timed JSONL files; use `aggregate_unified_results.py` with `**vendor/CD4dLLM`** (ETH checker) if you need functional accuracy.
- **Constraint %:** mean of per-instance `timing.constraint_pct` (wall-time fraction attributed to constraint / mask work, as logged by the runner).

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv pip install -r requirements.txt
```

For LLaDA: `uv pip install -r requirements-llada.txt` (pinned `transformers`).

**GGBS / LAVE bench (Dgrammar tree):** install `**llguidance>=1.6`** and put **dgrammar** on `PYTHONPATH`, e.g. clone into `vendor/dgrammar` (see below).

## Run GGBS vs LAVE (Dgrammar bench)

From `vendor/dgrammar` with `constrained_diffusion` on `PYTHONPATH` (see `vendor/dgrammar/README.md`):

```bash
cd vendor/dgrammar
# seed, limit, dataset, diffusion steps, offset
python bench/run_lave_ggbs.py 0 272 jsonschema 128 0

# for LAVE baseline
python bench/run_lave_timed.py 0 272 jsonsch
```

Timed LAVE-only runs use `bench/run_lave_timed.py` with the same argument pattern. Outputs are JSONL under `vendor/dgrammar/results/` (e.g. `ggbs_lave_timed_jsonschema_s0_t128.jsonl`, `lave_timed_jsonschema_s0_t128.jsonl`).

## Project layout (core)

```
src/
├── ggbs.py              # GGBS: segmented beam, caches, suffix prune
├── diffusion_sdsd.py    # Diffusion + schema-guided + GGBS hooks
├── baseline_dingo.py, sparse_dingo.py, herding.py, speculative_tree.py
├── test_ggbs.py         # GGBS unit tests
└── test_dllm_sdsd.py    # LLaDA / Dream loading and logits
vendor/dgrammar/         # Dgrammar + LAVE + bench (run_lave_ggbs.py)
```

## Other scripts


| Script                     | Purpose                                                 |
| -------------------------- | ------------------------------------------------------- |
| `run_ablation.py`          | B1/B2/B3/SDSD ablations on JSON-Mode-Eval (Dream/LLaDA) |
| `run_unified_benchmark.py` | SDSD family vs Dgrammar metrics on JSON-Bench           |
| `run_lave_sdsd_compare.py` | Aligned LAVE vs SDSD/BiDi comparison                    |
| `debug_diffusion.py`       | One-instance Dgrammar vs SDSD sanity check (GPU)        |


## Tests

```bash
cd src && python test_sdsd.py
cd src && python test_ggbs.py
cd src && python test_dllm_sdsd.py --mock
```

## Clone baselines

```bash
git clone https://github.com/guan404ming/dgrammar.git vendor/dgrammar
git clone https://github.com/zhangyitonggg/CD4dLLM.git vendor/CD4dLLM   # ETH / functional eval
git clone https://github.com/eth-sri/constrained-diffusion.git
```

## References

- **LAVE** (Zhang et al., ACM 2026): Lookahead-then-Verify for diffusion CFG decoding  
- **DINGO** (Suresh et al., NeurIPS 2025): Constrained inference for diffusion LLMs  
- **Constrained Decoding** (Mündler et al., NeurIPS 2025 DL4C): CFG decoding for diffusion LLMs  
- **STATIC** (Su et al., 2026): Sparse transition trie for constrained decoding

Further detail: `docs/ablation.md`, `docs/lave_sdsd_compare.md`, `docs/experiment_comparison.md`.
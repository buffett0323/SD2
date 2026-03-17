# SDSD Ablation Experiment Design

## 1. Experimental Settings

### Model Backbones

| Model | Type | VRAM | Transformers |
|-------|------|------|--------------|
| **LLaDA-8B-Instruct** | Discrete Diffusion | ~16GB | 4.38.2 |
| **Dream-7B-Instruct** | Diffusion LLM | ~20GB | ≥4.46 |

### Hardware Environment

- **GPU**: 24GB+ VRAM recommended
- **Optimization**: CSR sparse indexing (O(K) transition cost); consider JAX/XLA static compilation for CSR operators

### Hyperparameters

| Parameter | Symbol | Suggested Value | Description |
|-----------|--------|-----------------|-------------|
| Diffusion steps | $T$ | 128, 256, 512 | Discrete diffusion steps (LLaDA) |
| Speculative block size | $\gamma$ | 8, 16 | Speculative Tree draft length |
| Decoding temperature | Temp | 0.0 (Argmax), 0.4 | Common for quality evaluation |
| Herding delay | $\delta$ | 0.0005 ~ 0.01 | Momentum decay (if tunable) |

---

## 2. Evaluation Metrics

### Efficiency Metrics

| Metric | Description | Expected |
|--------|-------------|----------|
| **TTFT (ms)** | Time to First Token; startup latency after CSR eliminates O(N) scan | STATIC significantly lower |
| **NFE** | Number of Function Evaluations; target model forward pass count | Spec-Tree reduces 40–65% |
| **Throughput (tok/s)** | Tokens generated per second | Stable with STATIC |

#### NFE Measurement Semantics

- **Sequential methods (Baseline, Ablation 1, Ablation 2)**: One `model.forward()` call per generated token → **NFE = block_length**.
- **Block methods (Ablation 3, SDSD)**: One `model.forward()` for full-block logits, then tree decode on CPU → **NFE = T/τ** (rounds × 2 for draft+verify).
- Model inference must be included in the loop to correctly reflect SDSD’s NFE advantage; passing precomputed `prob_vectors` alone yields NFE = 1 for all methods.

### Reliability Metrics

| Metric | Description |
|--------|-------------|
| **Parse Rate (%)** | Fraction of outputs that conform to JSON/CFG grammar and parse successfully |

### Quality & Intent Metrics

| Metric | Description |
|--------|-------------|
| **Gen PPL** | Generated text perplexity |
| **Intent Recovery (steps)** | Steps to recover high-probability token to Top-1 after artificial perturbation |
| **Pass@1 (%)** | Functional correctness for code generation (HumanEval/MBPP) |

---

## 3. Datasets

| Dataset | Purpose | Hugging Face ID |
|---------|---------|-----------------|
| **JSON-Mode-Eval** | Complex nested JSON schema compliance | `NousResearch/json-mode-eval` |
| **HumanEval** | Code generation (CFG) | `openai/openai_humaneval` |
| **MBPP** | Python code generation | `google-research-datasets/mbpp` |
| **GSM-Symbolic** | Symbolic math reasoning structure stability | `apple/GSM-Symbolic` |

### Dataset Download

See “Dataset Download Guide” below.

---

## 4. Ablation Comparison Table

### LLaDA-8B-Instruct

*JSON-Mode-Eval, 20 samples, 64 tokens, vocab 126k*

| Method | Technique | Complexity | TTFT (ms) | Throughput (tok/s) | NFE (avg) | Parse Rate (%) | Pass@1 (%) | Intent Recovery (steps) |
|--------|-----------|------------|-----------|--------------------|-----------|----------------|-----------|--------------------------|
| **Baseline** | Original DINGO (O(N)) | O(N) | 10,985 | 5.83 | 64 | 100% | — | N/A |
| **Ablation 1** | STATIC + DINGO | O(K) | 7,614 | 8.41 | 64 | 100% | — | N/A |
| **Ablation 2** | DINGO + Herding | O(N) | 8,136 | 7.87 | 64 | 100% | — | 36.2 |
| **Ablation 3** | STATIC + Spec-Tree | O(K) | 2,110 | 30.33 | **8** | 100% | — | N/A |
| **Ours (SDSD)** | **STATIC + Herding + Tree** | **O(K)** | **2,673** | **23.94** | **8** | **100%** | — | 36.2 |

### Dream-7B-Instruct

*JSON-Mode-Eval, 20 samples, 64 tokens, vocab 151k*

| Method | Technique | Complexity | TTFT (ms) | Throughput (tok/s) | NFE (avg) | Parse Rate (%) | Pass@1 (%) | Intent Recovery (steps) |
|--------|-----------|------------|-----------|--------------------|-----------|----------------|-----------|--------------------------|
| **Baseline** | Original DINGO (O(N)) | O(N) | 11,448 | 5.59 | 64 | 100% | — | N/A |
| **Ablation 1** | STATIC + DINGO | O(K) | 7,320 | 8.74 | 64 | 100% | — | N/A |
| **Ablation 2** | DINGO + Herding | O(N) | 7,937 | 8.06 | 64 | 100% | — | 14.4 |
| **Ablation 3** | STATIC + Spec-Tree | O(K) | 2,235 | 28.64 | **8** | 100% | — | N/A |
| **Ours (SDSD)** | **STATIC + Herding + Tree** | **O(K)** | **2,913** | **21.97** | **8** | **100%** | — | 14.4 |

---

## 5. Experiment Logic Checklist

1. **STATIC**: TTFT drops significantly and throughput stays stable for large $N$ → validates $O(N) \to O(K)$.
2. **Herding**: Intent Recovery metric. SDSD should recover intended tokens faster via weight vector $w$.
3. **Speculative Tree**: NFE reduced by 40–65%+ → validates speculative parallel verification in structured settings.

---

## 6. Method–Code Mapping

| Method | Code Module | Function |
|--------|-------------|----------|
| Baseline | `baseline_dingo.py` | `baseline_dingo_dp` |
| Ablation 1 | `sparse_dingo.py` | `sparse_dingo_dp` |
| Ablation 2 | `herding.py` | `herding_decode` |
| Ablation 3 | `speculative_tree.py` | `speculative_decode_argmax` |
| SDSD | `speculative_tree.py` + `herding.py` | `speculative_decode` (herding path) |

## 7. How to Run

```bash
# Mock mode (no GPU, synthetic logits)
python run_ablation.py --model dream --mock --samples 20

# Dream-7B
python run_ablation.py --model dream --samples 20

# LLaDA-8B (requires transformers==4.38.2)
python run_ablation.py --model llada --samples 20

# With dataset
python run_ablation.py --model dream --dataset json-mode-eval --dataset-limit 50

# Custom output path
python run_ablation.py --model dream --output results/ablation_table.json
```

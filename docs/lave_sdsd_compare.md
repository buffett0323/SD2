# SDSD / BiDi vs LAVE：對齊實驗設定

本專案用 **`run_lave_sdsd_compare.py`** 在同一套設定下跑「我們的 diffusion + constraint」與 **LAVE**（若已安裝 `vendor/dgrammar`），並可選用 **`aggregate_unified_results.py`** 產生與 Dgrammar 風格一致的表格。

## 對齊了什麼

| 項目 | 值 |
|------|-----|
| 模型 | LLaDA-8B-Instruct（`run_unified_benchmark.py` 內載入） |
| 資料集 | Hugging Face `eth-sri/json-mode-eval-extended`（jsonschema / JSON-Bench） |
| Diffusion 步數 | `steps=128`（與 `run_unified_benchmark.sh` 裡 LAVE 的第三個維度一致） |
| Block / 長度 | `block_length=32`, `gen_length=256` |
| 溫度 | `generate_diffusion_sdsd` 內 `temperature=0.2`（兩邊若要比較需確認 LAVE 端是否同溫度；以 vendor 腳本為準） |

LAVE 由 **`vendor/dgrammar/bench/run_lave_timed.py`** 執行，參數慣例與 `run_unified_benchmark.sh` 相同：

```bash
python bench/run_lave_timed.py <seed> <num_instances> jsonschema <T> <extra>
```

## 前置：LAVE / Dgrammar

1. 依 [README.md](../README.md)「Cloning Baseline Repos」取得 `constrained-diffusion`、`CD4dLLM`、**Dgrammar** 等到 `vendor/`。
2. 依 `vendor/dgrammar` 的 README 安裝依賴（含 `llguidance` 等）。
3. 確認存在：`vendor/dgrammar/bench/run_lave_timed.py`。

沒有 vendor 時仍可只跑我們的方法：

```bash
python run_lave_sdsd_compare.py --methods sdsd,bidi --limit 20 --output results/lave_sdsd_compare
```

## 一鍵：我們 + LAVE + 聚合表

```bash
python run_lave_sdsd_compare.py \
  --methods sdsd,bidi \
  --limit 272 \
  --output results/lave_sdsd_compare \
  --run-lave \
  --aggregate
```

產物：

- 我們：`results/lave_sdsd_compare/sdsd_<method>_jsonschema.jsonl`
- LAVE：通常寫在 `vendor/dgrammar/results/`（依該 repo 為準）
- 表：`results/lave_sdsd_compare/unified_comparison.json`（並印出終端表格）

**ETH syntactic / functional** 需要 `vendor/CD4dLLM`（或 aggregate 能找到的 checker 路徑），否則相關欄位可能為 0。

## 只重新聚合已有結果

```bash
python run_lave_sdsd_compare.py --aggregate-only \
  --our-results results/lave_sdsd_compare \
  --lave-results vendor/dgrammar/results
```

等同：

```bash
python aggregate_unified_results.py results/lave_sdsd_compare vendor/dgrammar/results
```

## 與 `run_unified_benchmark.sh` 的關係

- `run_unified_benchmark.sh`：分別呼叫 `run_unified_benchmark.py`、Dgrammar、LAVE、IG-CD，再 aggregate。
- `run_lave_sdsd_compare.py`：**明確列出與 LAVE 對齊的 config**，並以單一入口跑「我們 + 可選 LAVE + 可選 aggregate」，適合寫進論文「同一模型、同一資料、同一 T」的實驗小節。

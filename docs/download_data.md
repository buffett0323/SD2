# 數據集下載指南 (Dataset Download Guide)

## 前置需求

```bash
pip install datasets
# 或
uv pip install datasets
```

## 方式一：Python 腳本下載

```bash
# 下載全部數據集
python scripts/download_datasets.py

# 指定緩存目錄
python scripts/download_datasets.py --cache-dir /path/to/cache

# 只下載部分
python scripts/download_datasets.py --datasets json-mode-eval,humaneval
```

## 方式二：在代碼中加載（自動下載）

```python
from datasets import load_dataset

# JSON-Mode-Eval
ds = load_dataset("NousResearch/json-mode-eval", split="train")

# HumanEval
ds = load_dataset("openai/openai_humaneval", split="test")

# MBPP
ds = load_dataset("google-research-datasets/mbpp", split="test")

# GSM-Symbolic
ds_main = load_dataset("apple/GSM-Symbolic", "main", split="test")
ds_p1 = load_dataset("apple/GSM-Symbolic", "p1", split="test")
ds_p2 = load_dataset("apple/GSM-Symbolic", "p2", split="test")
```

## 方式三：使用本專案 Dataset Loader

```bash
# 確保在專案 src 目錄或已將 src 加入 PYTHONPATH
cd src
python -c "
from dataset_loaders import get_dataset
samples = get_dataset('json-mode-eval')
samples = get_dataset('humaneval')
samples = get_dataset('mbpp')
samples = get_dataset('gsm-symbolic')  # config='main' 為預設
"
```

## 數據集詳情

| 數據集 | Hugging Face ID | 大小 | 用途 |
|--------|-----------------|------|------|
| JSON-Mode-Eval | `NousResearch/json-mode-eval` | ~81KB | JSON Schema 合規 |
| HumanEval | `openai/openai_humaneval` | ~84KB | 程式碼生成 |
| MBPP | `google-research-datasets/mbpp` | ~1MB | Python 程式碼 |
| GSM-Symbolic | `apple/GSM-Symbolic` | 多配置 | 符號數學推理 |

## 環境變數（可選）

```bash
# 加速下載（需 Hugging Face 帳號）
export HF_TOKEN=your_token

# 自訂緩存路徑
export HF_DATASETS_CACHE=/path/to/cache
```

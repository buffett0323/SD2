"""
Dataset loaders for SDSD ablation experiments.

Datasets:
  - JSON-Mode-Eval: NousResearch/json-mode-eval (JSON schema compliance)
  - HumanEval: openai/openai_humaneval (code generation)
  - MBPP: google-research-datasets/mbpp (Python code generation)
  - GSM-Symbolic: apple/GSM-Symbolic (math reasoning structure)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class EvalSample:
    """Single evaluation sample. prompt can be str or list of chat messages."""
    prompt: str | list  # str for simple prompts, list[dict] for chat format (json-mode-eval)
    schema_or_constraint: str | None = None
    expected_output: str | None = None
    dataset: str = ""
    id: str = ""


def load_json_mode_eval(split: str = "train", cache_dir: str | Path | None = None) -> list[EvalSample]:
    """
    Load JSON-Mode-Eval (NousResearch/json-mode-eval).
    prompt is a list of messages: [{"role":"system","content":"..."},{"role":"user","content":"..."}]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("NousResearch/json-mode-eval", split=split, cache_dir=str(cache_dir) if cache_dir else None)
    samples = []
    for i, row in enumerate(ds):
        prompt = row.get("prompt", row.get("system_prompt", "")) or ""
        schema = row.get("schema", row.get("expected", ""))
        # json-mode-eval uses prompt as list of messages; keep as-is for apply_chat_template
        samples.append(EvalSample(
            prompt=prompt if isinstance(prompt, list) else str(prompt),
            schema_or_constraint=str(schema) if schema else None,
            dataset="json-mode-eval",
            id=f"json_{i}",
        ))
    return samples


def load_humaneval(split: str = "test", cache_dir: str | Path | None = None) -> list[EvalSample]:
    """
    Load HumanEval for code generation (CFG compliance).
    Uses openai/openai_humaneval (164 Python problems).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("openai/openai_humaneval", split=split, cache_dir=str(cache_dir) if cache_dir else None)

    samples = []
    for i, row in enumerate(ds):
        prompt = row.get("prompt", row.get("instruction", "")) or ""
        if not prompt and "canonical_solution" in row:
            prompt = row.get("entry_point", "") or f"Complete the function: {row.get('task_id', '')}"
        samples.append(EvalSample(
            prompt=str(prompt),
            expected_output=row.get("canonical_solution", row.get("solution", "")),
            dataset="humaneval",
            id=str(row.get("task_id", f"humaneval_{i}")),
        ))
    return samples


def load_mbpp(split: str = "test", cache_dir: str | Path | None = None) -> list[EvalSample]:
    """
    Load MBPP (Mostly Basic Python Problems).
    Uses google-research-datasets/mbpp (parquet-based, no deprecated scripts).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("google-research-datasets/mbpp", split=split, cache_dir=str(cache_dir) if cache_dir else None)
    samples = []
    for i, row in enumerate(ds):
        prompt = row.get("text", row.get("task", "")) or ""
        samples.append(EvalSample(
            prompt=str(prompt),
            expected_output=row.get("code", ""),
            dataset="mbpp",
            id=row.get("task_id", f"mbpp_{i}"),
        ))
    return samples


def load_gsm_symbolic(config: str = "main", split: str = "test", cache_dir: str | Path | None = None) -> list[EvalSample]:
    """
    Load GSM-Symbolic (apple/GSM-Symbolic).
    config: "main", "p1", "p2"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    ds = load_dataset("apple/GSM-Symbolic", config, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    samples = []
    for i, row in enumerate(ds):
        prompt = row.get("question", row.get("modified_question", "")) or ""
        samples.append(EvalSample(
            prompt=str(prompt),
            expected_output=row.get("answer", row.get("modified_answer", "")),
            dataset="gsm-symbolic",
            id=f"gsm_{config}_{i}",
        ))
    return samples


def get_dataset(name: str, **kwargs) -> list[EvalSample]:
    """Load dataset by name."""
    if name == "json-mode-eval":
        return load_json_mode_eval(**kwargs)
    if name == "humaneval":
        return load_humaneval(**kwargs)
    if name == "mbpp":
        return load_mbpp(**kwargs)
    if name == "gsm-symbolic":
        return load_gsm_symbolic(config=kwargs.pop("config", "main"), **kwargs)
    raise ValueError(f"Unknown dataset: {name}. Choose: json-mode-eval, humaneval, mbpp, gsm-symbolic")

"""
JSONSchemaBench (arXiv:2501.10868) via Hugging Face ``epfl-dlab/JSONSchemaBench``.

This is **not** the same as ``register_dataset("jsonschema", ...)``, which loads
``eth-sri/json-mode-eval-extended`` and uses precomputed Lark grammars under
``cfgs_our/json/*.lark``.

JSONSchemaBench rows only provide ``json_schema`` and ``unique_id`` (no task-specific
user ``input`` / gold ``output``). We use a fixed instruction string so the LLaDA
prompt format matches the rest of the CD4dLLM eval stack.

Grammar for LAVE / llguidance: ``Checker`` accepts a canonical JSON Schema string;
see ``dgrammar.grammar_cache.get_cached_grammar`` (``grammar_from_json_schema``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterator

from datasets import concatenate_datasets, load_dataset

from constrained_diffusion.eval.dllm.datasets.generic import DataSet, Instance

if TYPE_CHECKING:
    from rustformlang.cfg import CFG
    from rustformlang.fa.dfa import DFA

# Matches the intent of JSON-mode / schema-constrained generation when the benchmark
# does not ship a per-schema natural-language task.
DEFAULT_JSONSCHEMABENCH_USER_PROMPT = (
    "Generate a single JSON value that conforms to the JSON schema given in the system message."
)


def _as_schema_str(json_schema: Any) -> str:
    if isinstance(json_schema, str):
        return json_schema
    return json.dumps(json_schema, ensure_ascii=False)


def _merge_splits(ds_dict) -> Any:
    """Merge train/val/test from a DatasetDict when present."""
    names = [n for n in ("train", "val", "test") if n in ds_dict]
    if names:
        return concatenate_datasets([ds_dict[n] for n in names])
    # Fallback: single split
    keys = list(ds_dict.keys())
    return ds_dict[keys[0]]


class JSONSchemaBenchInstance(Instance):
    def __init__(self, row: dict[str, Any]):
        self._instance_id = str(row["unique_id"])
        schema_str = _as_schema_str(row["json_schema"])
        self._schema_obj = json.loads(schema_str)
        self.data = {
            "instance_id": self._instance_id,
            "input": DEFAULT_JSONSCHEMABENCH_USER_PROMPT,
            "output": "",
            "schema": schema_str,
        }

    def instance_id(self) -> str:
        return self._instance_id

    def user_prompt_content(self) -> str:
        return self.data["input"]

    def language_short_name(self) -> str:
        return "json"

    def system_message_content(self) -> str:
        return (
            "You are a helpful assistant that answers in JSON. Here's the JSON schema you must adhere to:\n"
            f"<schema>\n{self.data['schema']}\n</schema>\n"
        )

    def language_lex_subtokens(
        self,
    ) -> tuple[CFG, dict[str, str | DFA], dict[str, set[str]]]:
        from constrained_diffusion.cfgs.jsonschema import schema_to_cfg
        return schema_to_cfg(self._schema_obj)

    def cfg(self) -> str:
        """Canonical JSON Schema string for llguidance (not a precomputed Lark file)."""
        return json.dumps(self._schema_obj, sort_keys=True, ensure_ascii=False)


class JSONSchemaBenchDataSet(DataSet):
    """
    Load `epfl-dlab/JSONSchemaBench <https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench>`_.

    Parameters
    ----------
    subset:
        Config name, e.g. ``\"default\"`` (all), ``\"Github_easy\"``, ``\"Kubernetes\"``, …
        See the dataset card for the current list of configs.
    """

    def __init__(self, subset: str = "default"):
        super().__init__()
        self.subset = subset
        self._rows = None
        self.different_grammar_per_instance = True

    def load_data(self):
        if self._rows is None:
            raw = load_dataset(
                "epfl-dlab/JSONSchemaBench",
                self.subset,
            )
            self._rows = _merge_splits(raw)
        return self._rows

    def __iter__(self) -> Iterator[Instance]:
        for row in self.load_data():
            yield JSONSchemaBenchInstance(row)


def register_jsonschemabench_alias(registry_name: str, hf_subset: str) -> None:
    """
    Register another ``epfl-dlab/JSONSchemaBench`` config under ``registry_name``.

    Call **before** ``load_dataset(registry_name)`` from ``eval.dllm.dataset``::

        from constrained_diffusion.eval.dllm.datasets.jsonschemabench_hf import (
            register_jsonschemabench_alias,
        )
        register_jsonschemabench_alias(\"jsonschemabench_github_easy\", \"Github_easy\")

    Config names match the Hugging Face dataset card (e.g. ``\"Github_easy\"``, ``\"Kubernetes\"``).
    """
    from constrained_diffusion.eval.dllm.dataset import register_dataset

    register_dataset(registry_name, JSONSchemaBenchDataSet(hf_subset))

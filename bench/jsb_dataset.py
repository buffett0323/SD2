"""JSONSchemaBench dataset adapter for dgrammar/LAVE benchmark runners.

Loads schemas from epfl-dlab/JSONSchemaBench and wraps them as
constrained_diffusion Instance/DataSet objects so the existing
runners can use them without modification.

Usage:
    import jsb_dataset  # registers jsb_hard, jsb_medium, etc.
    from constrained_diffusion.eval.dllm.dataset import load_dataset
    dataset = load_dataset("jsb_hard")  # test split only; use jsb_hard_all for train+val+test
"""

from typing import Iterator

from datasets import load_dataset as hf_load_dataset, concatenate_datasets

from constrained_diffusion.eval.dllm.datasets.generic import DataSet, Instance
from constrained_diffusion.eval.dllm.dataset import register_dataset

try:
    from constrained_diffusion.cfgs.jsonschema import schema_to_cfg
    from rustformlang.cfg import CFG
    from rustformlang.fa.dfa import DFA

    HAS_RUSTFORMLANG = True
except ImportError:
    HAS_RUSTFORMLANG = False

import json


class JSBInstance(Instance):
    """Single JSONSchemaBench instance."""

    def __init__(self, unique_id: str, json_schema: str):
        self._instance_id = unique_id
        self.data = {"schema": json_schema}

    def instance_id(self) -> str:
        return self._instance_id

    def user_prompt_content(self) -> str:
        return "Generate a valid JSON object that matches the provided schema."

    def language_short_name(self) -> str:
        return "json"

    def system_message_content(self) -> str:
        return (
            "You are a helpful assistant that answers in JSON. "
            "Here's the JSON schema you must adhere to:\n<schema>\n"
            f"{self.data['schema']}\n</schema>\n"
        )

    def extract_result(self, s: str) -> str:
        from constrained_diffusion.eval.dllm.datasets.generic import extract_code
        return extract_code(s, "json", 0)

    def cfg(self) -> str:
        """Return JSON schema string for the CD4dLLM Checker (llguidance JSON Schema).

        Unlike the eth-sri ``jsonschema`` dataset (which uses per-instance ``.lark``
        files via ``get_cfg``), this passes the raw Hugging Face schema. Some
        JSONSchemaBench rows may log validator warnings; the Checker still builds.
        """
        return self.data["schema"]

    def language_lex_subtokens(self):
        if not HAS_RUSTFORMLANG:
            raise RuntimeError("rustformlang not available for LAVE cfg conversion")
        return schema_to_cfg(json.loads(self.data["schema"]))


class JSBDataSet(DataSet):
    """Wraps a JSONSchemaBench subset as a DataSet."""

    def __init__(self, subset: str, split: str = "test"):
        super().__init__()
        self.subset = subset
        self.split = split
        self.different_grammar_per_instance = True
        self._data = None

    def _load(self):
        if self._data is not None:
            return self._data
        ds = hf_load_dataset("epfl-dlab/JSONSchemaBench", name=self.subset)
        if self.split == "all":
            parts = [ds[s] for s in ds if len(ds[s]) > 0]
            self._data = concatenate_datasets(parts)
        else:
            self._data = ds[self.split]
        return self._data

    def __iter__(self) -> Iterator[Instance]:
        for row in self._load():
            yield JSBInstance(row["unique_id"], row["json_schema"])


# Register datasets on import
_SUBSETS = {
    "jsb_hard": "Github_hard",
    "jsb_medium": "Github_medium",
    "jsb_easy": "Github_easy",
    "jsb_ultra": "Github_ultra",
    "jsb_trivial": "Github_trivial",
    "jsb_glaive": "Glaiveai2K",
    "jsb_k8s": "Kubernetes",
    "jsb_store": "JsonSchemaStore",
    "jsb_snowplow": "Snowplow",
    "jsb_wapo": "WashingtonPost",
}

for name, subset in _SUBSETS.items():
    for reg_name, split in [
        (name, "test"),
        (f"{name}_all", "all"),
        (f"{name}_test", "test"),
        (f"{name}_train", "train"),
        (f"{name}_val", "val"),
    ]:
        try:
            register_dataset(reg_name, JSBDataSet(subset, split=split))
        except ValueError:
            pass  # already registered

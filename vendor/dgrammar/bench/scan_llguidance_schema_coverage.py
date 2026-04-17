#!/usr/bin/env python3
"""
Scan JSONSchemaBench (or any HF config) with llguidance only — no LAVE, no model.

For each instance:
  1. Normalize ``json_schema`` to a string.
  2. ``LLMatcher.grammar_from_json_schema`` (may raise).
  3. ``LLMatcher.validate_grammar_with_warnings`` → ``is_err``, warning strings.
  4. Optionally walk the JSON schema object to flag surface keys: oneOf, not, …

Outputs JSONL lines you can aggregate to see which unique_id / patterns break or warn.

Usage (from ``final/vendor/dgrammar`` with env that has ``datasets`` + ``llguidance``)::

    python bench/scan_llguidance_schema_coverage.py --subset Github_hard --out results/llguidance_audit_jsb_hard.jsonl

    python bench/scan_llguidance_schema_coverage.py --subset Github_hard --limit 50 --out /tmp/sample.jsonl

    python bench/scan_llguidance_schema_coverage.py --registry-name jsb_hard

``--registry-name jsb_hard`` maps to HF subset ``Github_hard`` (same as ``jsb_dataset._SUBSETS``).
Default ``--split`` is **test** only; use ``--split all`` for train+val+test.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

from datasets import concatenate_datasets, load_dataset
from llguidance import LLMatcher


# Mirrors bench/jsb_dataset.py _SUBSETS (HF config name)
REGISTRY_TO_SUBSET: dict[str, str] = {
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


@dataclass
class SchemaKeyFlags:
    """Presence of selected keywords anywhere in the JSON schema object."""

    has_oneOf: bool = False
    has_anyOf: bool = False
    has_allOf: bool = False
    has_not: bool = False
    has_if: bool = False
    has_dependencies: bool = False
    has_patternProperties: bool = False
    has_additionalProperties_false: bool = False


def _walk_schema(obj: Any, flags: SchemaKeyFlags) -> None:
    if isinstance(obj, dict):
        if "oneOf" in obj:
            flags.has_oneOf = True
        if "anyOf" in obj:
            flags.has_anyOf = True
        if "allOf" in obj:
            flags.has_allOf = True
        if "not" in obj:
            flags.has_not = True
        if "if" in obj:
            flags.has_if = True
        if "dependencies" in obj or "dependentSchemas" in obj:
            flags.has_dependencies = True
        if "patternProperties" in obj:
            flags.has_patternProperties = True
        if obj.get("additionalProperties") is False:
            flags.has_additionalProperties_false = True
        for v in obj.values():
            _walk_schema(v, flags)
    elif isinstance(obj, list):
        for x in obj:
            _walk_schema(x, flags)


def _normalize_schema_string(raw: str | dict | Any) -> str:
    if isinstance(raw, str):
        # Ensure valid JSON (may already be a stringified schema)
        return raw
    return json.dumps(raw, ensure_ascii=False)


def _categorize_warnings(warn_msgs: list[str]) -> dict[str, bool]:
    text = " ".join(warn_msgs).lower()
    return {
        "warn_oneOf": "oneof" in text,
        "warn_coerce_oneOf": "coerce_one_of" in text or "coerce_oneof" in text,
        "warn_unsatisfiable": "unsatisfiable" in text,
        "warn_incompatible_types": "incompatible types" in text,
        "warn_unimplemented_not": "unimplemented" in text and "not" in text,
        "warn_unimplemented_keys": "unimplemented keys" in text,
    }


def _audit_one(schema_str: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "grammar_from_json_schema_ok": False,
        "grammar_build_exception": None,
        "validate_is_err": None,
        "validate_warnings": None,
    }

    try:
        obj = json.loads(schema_str)
    except json.JSONDecodeError as e:
        row["grammar_build_exception"] = f"JSONDecodeError: {e}"
        return row

    flags = SchemaKeyFlags()
    _walk_schema(obj, flags)
    row["schema_key_flags"] = asdict(flags)

    try:
        grm = LLMatcher.grammar_from_json_schema(schema_str)
    except Exception as e:
        row["grammar_build_exception"] = f"{type(e).__name__}: {e}"
        return row

    row["grammar_from_json_schema_ok"] = True
    is_err, warn_msgs = LLMatcher.validate_grammar_with_warnings(grm)
    row["validate_is_err"] = bool(is_err)
    row["validate_warnings"] = list(warn_msgs)
    row["warning_categories"] = _categorize_warnings(warn_msgs)
    return row


def _load_rows(subset: str, split: str):
    ds = load_dataset("epfl-dlab/JSONSchemaBench", name=subset)
    if split == "all":
        parts = [ds[s] for s in ds if len(ds[s]) > 0]
        return concatenate_datasets(parts)
    return ds[split]


def main() -> None:
    p = argparse.ArgumentParser(description="llguidance-only JSON Schema audit for JSONSchemaBench")
    p.add_argument(
        "--subset",
        default=None,
        help="HF config name, e.g. Github_hard (default: from --registry-name or Github_hard)",
    )
    p.add_argument(
        "--registry-name",
        default=None,
        help=f"Shortcut: one of {list(REGISTRY_TO_SUBSET.keys())} → HF subset",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Dataset split: all | train | val | test (default: test)",
    )
    p.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all)")
    p.add_argument("--out", "-o", required=True, help="Output JSONL path")
    p.add_argument("--progress-every", type=int, default=200, help="Print progress every N rows")
    args = p.parse_args()

    if args.registry_name:
        if args.registry_name not in REGISTRY_TO_SUBSET:
            sys.exit(f"Unknown --registry-name {args.registry_name!r}. Choose from {sorted(REGISTRY_TO_SUBSET)}")
        subset = REGISTRY_TO_SUBSET[args.registry_name]
    elif args.subset:
        subset = args.subset
    else:
        subset = "Github_hard"

    rows = _load_rows(subset, args.split)
    n_total = len(rows)
    to_process = n_total if args.limit <= 0 else min(args.limit, n_total)

    print(f"subset={subset!r} split={args.split!r} rows={n_total} processing={to_process}", flush=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(to_process):
            if args.progress_every and i > 0 and i % args.progress_every == 0:
                print(f"  ... {i}/{to_process}", flush=True)
            row = rows[i]
            uid = row["unique_id"]
            raw_schema = row["json_schema"]
            if not isinstance(raw_schema, str):
                schema_str = json.dumps(raw_schema, ensure_ascii=False)
            else:
                schema_str = raw_schema

            audit = _audit_one(schema_str)
            out = {
                "unique_id": uid,
                "subset": subset,
                "split": args.split,
                **audit,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} lines to {args.out}", flush=True)

    # Tiny stdout summary from the file we wrote
    summary: dict[str, int] = {}
    build_fail = 0
    validate_err = 0
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if not o.get("grammar_from_json_schema_ok"):
                build_fail += 1
            elif o.get("validate_is_err"):
                validate_err += 1
            for k, v in (o.get("warning_categories") or {}).items():
                if v:
                    summary[k] = summary.get(k, 0) + 1

    print(
        f"Summary: grammar_from_json_schema failed: {build_fail}/{written}, "
        f"validate_is_err: {validate_err}/{written}",
        flush=True,
    )
    if summary:
        print("Warning category counts (rows where flag True):", flush=True)
        for k in sorted(summary.keys()):
            print(f"  {k}: {summary[k]}", flush=True)


if __name__ == "__main__":
    main()

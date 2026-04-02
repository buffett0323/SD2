#!/usr/bin/env python3
"""
Aggregate LAVE / GGBS JSONL runs on JSONSchemaBench into README-style metrics.

Fills the same columns as ``vendor/dgrammar/README.md`` (Main Comparison table):

  | Method | Syntactic | Functional | Mean Time | Median | P95 | Max | Constraint % |

**Syntactic / Functional** (JSONSchemaBench, no ETH test suite in-repo):

- **Syntactic**: share of rows where ``extracted`` parses as JSON and validates against
  the row's ``schema`` (JSON Schema string), using :mod:`jsonschema` with the appropriate
  draft validator.
- **Functional**: same as syntactic here — the benchmark rows do not ship gold outputs or
  JSON Schema Test Suite cases in our JSONL. For json-mode-eval (272 instances), use
  ``aggregate_unified_results.py`` with ``vendor/CD4dLLM`` for ETH ``passed_tests``.

**Diffusion validity** (EOS/mask, field ``valid`` in JSONL) is reported separately as
``diffusion_valid_pct``.

Usage::

    cd vendor/dgrammar
    python bench/jsonschemabench_metrics.py results/lave_timed_jsonschemabench_s0_t128.jsonl

    # multiple files (e.g. chunks merged manually)
    python bench/jsonschemabench_metrics.py results/a.jsonl results/b.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# jsonschema is a dgrammar dependency (pyproject.toml)
from jsonschema import validators
from jsonschema.exceptions import ValidationError


def _percentile_sorted(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def validate_instance_against_schema(extracted: str | None, schema_str: str) -> bool:
    """True if ``extracted`` is JSON and validates under ``schema_str``."""
    if not extracted or not str(extracted).strip():
        return False
    try:
        instance = json.loads(extracted)
    except (json.JSONDecodeError, TypeError):
        return False
    try:
        schema = json.loads(schema_str)
    except (json.JSONDecodeError, TypeError):
        return False
    try:
        cls = validators.validator_for(schema)
        cls(schema).validate(instance)
        return True
    except (ValidationError, Exception):
        return False


def load_rows(paths: list[Path]) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                iid = d.get("instance_id", "")
                if iid in seen:
                    continue
                seen.add(iid)
                rows.append(d)
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Compute stats for rows that include ``schema`` (JSONSchemaBench JSONL)."""
    if not rows:
        return {}

    missing_schema = sum(1 for r in rows if not r.get("schema"))
    times: list[float] = []
    constraint_pcts: list[float] = []
    diffusion_ok = 0
    schema_ok = 0

    for r in rows:
        t = r.get("time_taken")
        if t is not None:
            times.append(float(t))
        timing = r.get("timing") or {}
        cp = timing.get("constraint_pct")
        if cp is not None:
            constraint_pcts.append(float(cp))

        if r.get("valid"):
            diffusion_ok += 1

        sch = r.get("schema")
        if sch:
            ext = r.get("extracted")
            if validate_instance_against_schema(ext, sch):
                schema_ok += 1

    n = len(rows)
    times.sort()
    constraint_avg = sum(constraint_pcts) / len(constraint_pcts) if constraint_pcts else 0.0

    return {
        "n": n,
        "missing_schema_count": missing_schema,
        "diffusion_valid_pct": diffusion_ok / n * 100 if n else 0.0,
        "syntactic_pct": schema_ok / n * 100 if n else 0.0,
        "functional_pct": schema_ok / n * 100 if n else 0.0,
        "mean_time_s": sum(times) / len(times) if times else 0.0,
        "median_time_s": _percentile_sorted(times, 0.5),
        "p95_time_s": _percentile_sorted(times, 0.95),
        "max_time_s": max(times) if times else 0.0,
        "constraint_pct_mean": constraint_avg,
    }


def print_report(stats: dict, method_label: str = "LAVE") -> None:
    if not stats:
        print("No statistics (empty input).")
        return
    n = stats["n"]
    print(f"\n{method_label} — JSONSchemaBench metrics (n={n})\n")
    if stats.get("missing_schema_count"):
        print(
            f"  Warning: {stats['missing_schema_count']} rows lack `schema`; "
            "syntactic/functional require `schema` in JSONL (re-run with updated bench)."
        )
    print(
        f"  Diffusion `valid` (EOS/mask): {stats['diffusion_valid_pct']:.1f}%\n"
        f"  Syntactic (JSON + schema valid): {stats['syntactic_pct']:.1f}%\n"
        f"  Functional (same as syntactic here): {stats['functional_pct']:.1f}%\n"
        f"  Mean time:   {stats['mean_time_s']:.2f}s\n"
        f"  Median time: {stats['median_time_s']:.2f}s\n"
        f"  P95 time:    {stats['p95_time_s']:.2f}s\n"
        f"  Max time:    {stats['max_time_s']:.2f}s\n"
        f"  Constraint % (mean): {stats['constraint_pct_mean']:.1f}%\n"
    )
    print("README row (markdown):\n")
    print(
        f"| {method_label} | {stats['syntactic_pct']:.1f}% | {stats['functional_pct']:.1f}% | "
        f"{stats['mean_time_s']:.2f}s | {stats['median_time_s']:.2f}s | "
        f"{stats['p95_time_s']:.2f}s | {stats['max_time_s']:.2f}s | "
        f"{stats['constraint_pct_mean']:.1f}% |\n"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:] if not p.startswith("-")]
    if not paths:
        print("No input files.")
        sys.exit(1)

    rows = load_rows(paths)
    stats = aggregate(rows)
    # Infer label from first filename
    tag = paths[0].stem.lower()
    if "lave" in tag:
        method = "LAVE"
    elif "ggbs" in tag:
        method = "GGBS"
    elif "igcd" in tag:
        method = "IG-CD"
    else:
        method = paths[0].stem
    print_report(stats, method_label=method)


if __name__ == "__main__":
    main()

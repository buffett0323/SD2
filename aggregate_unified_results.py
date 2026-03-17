#!/usr/bin/env python3
"""
Aggregate unified benchmark results into Dgrammar-style comparison table.

Reads result JSONL files from SDSD, Dgrammar, LAVE and produces:
  | Method | Syntactic | Functional | Mean Time | Median | P95 | Max | Constraint % |

Usage:
  python aggregate_unified_results.py [results_dir]
  python aggregate_unified_results.py results/unified
  python aggregate_unified_results.py vendor/dgrammar/results

Result files expected (JSONL, one JSON object per line):
  - instance_id, extracted, time_taken, method (or inferred from filename)
  - timing.constraint_pct (Dgrammar-style: time overhead %)
  - For SDSD: constraint_pct in our format (token validity) is stored but we use timing.constraint_pct if present
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# Method name mapping: filename tag -> display name
METHOD_NAMES = {
    "baseline": "Baseline (DINGO O(N))",
    "ablation1": "Ablation1 (STATIC+DINGO)",
    "ablation2": "Ablation2 (Herding)",
    "ablation3": "Ablation3 (Spec-Tree)",
    "sdsd": "SDSD (Ours)",
    "dgrammar": "Dgrammar",
    "dgrammar_v2_async": "Dgrammar v2+async+AC4",
    "lave": "LAVE",
    "lave_timed": "LAVE",
    "igcd": "IG-CD",
    "nocd": "NO-CD",
}


def load_checker():
    """Load ETH checker for jsonschema evaluation."""
    # Try vendor paths (CD4dLLM has eval/ and constrained_diffusion/)
    for base in [
        Path(__file__).resolve().parent / "vendor" / "CD4dLLM",
        Path(__file__).resolve().parent / "vendor" / "dgrammar" / "vendor" / "CD4dLLM",
    ]:
        if (base / "eval" / "dllm" / "jsonmode" / "checker.py").exists():
            sys.path.insert(0, str(base))
            break
    try:
        from eval.dllm.jsonmode.checker import check_instance
        return check_instance
    except ImportError as e:
        print(f"Warning: Could not load ETH checker: {e}")
        return None


def eval_results(check_fn, results: list[dict]) -> list[dict]:
    """Run ETH checker on each result. Add syntax_ok, passed_tests."""
    if not check_fn:
        for r in results:
            r["syntax_ok"] = False
            r["passed_tests"] = False
        return results

    for r in results:
        try:
            ev = check_fn(r, timeout=40)
            r["syntax_ok"] = ev.get("syntax_ok", False)
            r["passed_tests"] = ev.get("passed_tests", False)
        except Exception as e:
            r["syntax_ok"] = False
            r["passed_tests"] = False
            r["eval_error"] = str(e)
    return results


def load_result_files(results_dir: Path, method_filter: list[str] | None = None) -> dict[str, list[dict]]:
    """Load all JSONL result files, group by method."""
    method_results: dict[str, list[dict]] = {}

    # SDSD unified format: sdsd_baseline_jsonschema.jsonl, sdsd_ablation1_jsonschema.jsonl, ...
    # Dgrammar format: v2_async_ac4_timed_jsonschema_s0_t128.jsonl, lave_timed_jsonschema_s0_t128.jsonl
    patterns = [
        "*_jsonschema_*.jsonl",
        "*_jsonschema.jsonl",
        "sdsd_*_jsonschema.jsonl",
    ]

    for pattern in patterns:
        for f in results_dir.glob(pattern):
            if ".compiled." in f.name or ".merged." in f.name:
                continue

            # Infer method from filename
            name = f.stem.lower()
            method = "unknown"
            if "sdsd_baseline" in name:
                method = "baseline"
            elif "sdsd_ablation1" in name:
                method = "ablation1"
            elif "sdsd_ablation2" in name:
                method = "ablation2"
            elif "sdsd_ablation3" in name:
                method = "ablation3"
            elif "sdsd_sdsd" in name:
                method = "sdsd"
            elif ("v2_async" in name or "dgrammar" in name) and "lave" not in name:
                method = "dgrammar_v2_async"
            elif "lave" in name:
                method = "lave_timed"
            elif "igcd" in name:
                method = "igcd"
            elif "nocd" in name:
                method = "nocd"
            else:
                for tag in METHOD_NAMES:
                    if tag in name:
                        method = tag
                        break

            if method_filter and method not in method_filter:
                continue

            lines = []
            with open(f) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        d["_source_file"] = f.name
                        if "method" not in d:
                            d["method"] = method
                        lines.append(d)
                    except json.JSONDecodeError:
                        pass

            if lines:
                key = METHOD_NAMES.get(method, method)
                if key not in method_results:
                    method_results[key] = []
                # Deduplicate by instance_id
                seen = set()
                for d in lines:
                    iid = d.get("instance_id", "")
                    if iid not in seen:
                        seen.add(iid)
                        method_results[key].append(d)

    return method_results


def compute_stats(results: list[dict], check_fn) -> dict:
    """Compute Syntactic, Functional, Mean Time, Median, P95, Max, Constraint %."""
    if not results:
        return {}

    # Run eval if we have checker
    results = eval_results(check_fn, list(results))

    n = len(results)
    syntax_ok = sum(1 for r in results if r.get("syntax_ok", False))
    passed = sum(1 for r in results if r.get("passed_tests", False))

    times = []
    constraint_pcts = []
    for r in results:
        t = r.get("time_taken")
        if t is not None:
            times.append(float(t))
        timing = r.get("timing") or {}
        cp = timing.get("constraint_pct") or r.get("constraint_pct")
        if cp is not None:
            constraint_pcts.append(float(cp))

    times = sorted(times) if times else []
    avg_time = sum(times) / len(times) if times else 0
    median_time = times[len(times) // 2] if times else 0
    p95_time = times[int(len(times) * 0.95)] if len(times) >= 20 else (times[-1] if times else 0)
    max_time = max(times) if times else 0

    # constraint_pct: Dgrammar/LAVE use 0-100; ensure we output 0-100
    constraint_avg = sum(constraint_pcts) / len(constraint_pcts) if constraint_pcts else 0
    if constraint_pcts and max(constraint_pcts) <= 1:
        constraint_avg *= 100

    return {
        "n": n,
        "syntactic": syntax_ok / n * 100 if n else 0,
        "functional": passed / n * 100 if n else 0,
        "mean_time": avg_time,
        "median_time": median_time,
        "p95_time": p95_time,
        "max_time": max_time,
        "constraint_pct": constraint_avg,
    }


def print_table(stats: dict[str, dict]):
    """Print Dgrammar-style comparison table."""
    print("\n" + "=" * 120)
    print("  Unified Benchmark — JSON-Bench (jsonschema), LLaDA-8B-Instruct, T=128, L=256")
    print("=" * 120)
    print(f"{'Method':<28} {'Syntactic':<12} {'Functional':<12} {'Mean Time':<12} {'Median':<12} {'P95':<12} {'Max':<12} {'Constraint %':<12}")
    print("-" * 120)

    for method, s in sorted(stats.items(), key=lambda x: -x[1].get("syntactic", 0)):
        syn = s.get("syntactic", 0)
        func = s.get("functional", 0)
        mean_t = s.get("mean_time", 0)
        med_t = s.get("median_time", 0)
        p95_t = s.get("p95_time", 0)
        max_t = s.get("max_time", 0)
        cp = s.get("constraint_pct", 0)
        func_str = f"{func:.1f}%" if func > 0 else "-"
        print(f"{method:<28} {syn:.1f}%{'':<6} {func_str:<12} {mean_t:.2f}s{'':<6} {med_t:.2f}s{'':<6} {p95_t:.2f}s{'':<6} {max_t:.2f}s{'':<6} {cp:.1f}%")
    print("=" * 120 + "\n")


def main():
    base = Path(__file__).resolve().parent
    # Accept multiple result dirs: aggregate_unified_results.py [dir1] [dir2] ...
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        results_dirs = [Path(p) for p in sys.argv[1:] if not p.startswith("-")]
    else:
        results_dirs = [
            base / "results" / "unified",
            base / "vendor" / "dgrammar" / "results",
        ]

    print("Loading results from:", [str(d) for d in results_dirs])

    check_fn = load_checker()
    if not check_fn:
        print("ETH checker not found. Syntactic/Functional will be 0. Install vendor/CD4dLLM or vendor/dgrammar.")

    all_results: dict[str, list[dict]] = {}
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        for method, results in load_result_files(results_dir).items():
            if method not in all_results:
                all_results[method] = []
            # Dedupe by instance_id across dirs
            seen = {r.get("instance_id") for r in all_results[method]}
            for r in results:
                if r.get("instance_id") not in seen:
                    seen.add(r.get("instance_id"))
                    all_results[method].append(r)

    if not all_results:
        print("No result files found.")
        return 1

    # Filter methods if specified (e.g. --methods baseline,sdsd)
    method_filter = None
    for i, a in enumerate(sys.argv):
        if a == "--methods" and i + 1 < len(sys.argv):
            method_filter = [m.strip().lower() for m in sys.argv[i + 1].split(",")]
            break
    if method_filter:
        all_results = {
            k: v for k, v in all_results.items()
            if any(mf in k.lower() for mf in method_filter)
        }

    stats = {}
    for method, results in all_results.items():
        print(f"  {method}: {len(results)} instances")
        stats[method] = compute_stats(results, check_fn)

    print_table(stats)

    # Save to JSON
    out_dir = results_dirs[0] if results_dirs else base / "results" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "unified_comparison.json"
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

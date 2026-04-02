"""Evaluate benchmark results using ETH's checkers.

Reads results from ../results/ and produces syntax/functional correctness stats.
Supports Dgrammar, LAVE, and IG-CD result files.

Usage:
    python bench/eval_results.py                     # evaluate all found results
    python bench/eval_results.py v2_async_ac4_timed  # evaluate specific tag
    python bench/eval_results.py lave_timed          # evaluate LAVE results
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def eval_jsonschema(input_file: str, output_file: str):
    """Evaluate JSON results using ETH's checker."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "constrained-diffusion"))
    from eval.dllm.jsonmode.checker import check_instance

    results = []
    with open(input_file) as f:
        lines = f.readlines()

    with open(output_file, "w") as out:
        for line in lines:
            d = json.loads(line)
            try:
                result = check_instance(d, timeout=40)
                result["time_taken"] = d.get("time_taken")
                result["resamples"] = d.get("resamples")
                results.append(result)
                out.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"  Error checking {d.get('instance_id', '?')}: {e}")

    n = len(results)
    if n == 0:
        print(f"  No results in {input_file}")
        return None

    syntax_ok = sum(1 for r in results if r.get("syntax_ok", False))
    passed = sum(1 for r in results if r.get("passed_tests", False))
    times = [r.get("time_taken", 0) or 0 for r in results]
    avg_time = sum(times) / n

    print(f"  {n} instances, syntax={syntax_ok}/{n} ({syntax_ok/n*100:.1f}%), "
          f"functional={passed}/{n} ({passed/n*100:.1f}%), avg_time={avg_time:.1f}s")
    return {"n": n, "syntax_ok": syntax_ok, "passed": passed, "avg_time": avg_time}


def eval_jsonschemabench(input_file: str, output_file: str):
    """
    Evaluate JSONL from ``run_lave_timed`` / ``run_lave_ggbs`` on JSONSchemaBench
    (rows must include ``schema``). Sets ``syntax_ok`` / ``passed_tests`` via
    :mod:`jsonschema` instance validation (same notion as ``jsonschemabench_metrics``).
    """
    from jsonschemabench_metrics import validate_instance_against_schema

    results = []
    with open(input_file) as f:
        lines = f.readlines()

    with open(output_file, "w") as out:
        for line in lines:
            d = json.loads(line)
            sch = d.get("schema")
            ext = d.get("extracted")
            ok = bool(sch) and validate_instance_against_schema(ext, sch)
            d["syntax_ok"] = ok
            d["passed_tests"] = ok
            results.append(d)
            out.write(json.dumps(d) + "\n")

    n = len(results)
    if n == 0:
        print(f"  No results in {input_file}")
        return None

    syntax_ok = sum(1 for r in results if r.get("syntax_ok", False))
    passed = sum(1 for r in results if r.get("passed_tests", False))
    times = [r.get("time_taken", 0) or 0 for r in results]
    avg_time = sum(times) / n

    print(f"  {n} instances, syntax={syntax_ok}/{n} ({syntax_ok/n*100:.1f}%), "
          f"functional={passed}/{n} ({passed/n*100:.1f}%), avg_time={avg_time:.1f}s")
    return {"n": n, "syntax_ok": syntax_ok, "passed": passed, "avg_time": avg_time}


def find_result_files(tag_filter=None):
    """Find result JSONL files, optionally filtered by tag."""
    files = sorted(
        set(
            list(RESULTS_DIR.glob("*_jsonschema_*.jsonl"))
            + list(RESULTS_DIR.glob("*_jsonschemabench_*.jsonl"))
        )
    )
    # Exclude compiled/eval output files
    files = [f for f in files if ".compiled." not in f.name]
    if tag_filter:
        files = [f for f in files if tag_filter in f.name]
    return files


def merge_chunks(files):
    """Group chunked result files by tag+seed+steps, merge into single list."""
    groups = {}
    for f in files:
        # Strip _off{N} suffix to group chunks together
        name = f.stem
        # Remove _off{N} suffix
        base = name
        for part in name.split("_"):
            if part.startswith("off"):
                base = name[:name.index(f"_{part}")]
                break
        if base not in groups:
            groups[base] = []
        groups[base].append(f)
    return groups


def main():
    tag_filter = sys.argv[1] if len(sys.argv) > 1 else None

    files = find_result_files(tag_filter)
    if not files:
        print(f"No result files found in {RESULTS_DIR}")
        if tag_filter:
            print(f"  (filter: {tag_filter})")
        return

    groups = merge_chunks(files)

    for base_name, chunk_files in sorted(groups.items()):
        print(f"\n{base_name} ({len(chunk_files)} file(s)):")

        # Merge all chunks, deduplicate by instance_id
        seen = set()
        all_lines = []
        for f in sorted(chunk_files):
            with open(f) as fh:
                for line in fh:
                    d = json.loads(line)
                    iid = d.get("instance_id", "")
                    if iid not in seen:
                        seen.add(iid)
                        all_lines.append(line)

        # Write merged file for eval
        merged_file = RESULTS_DIR / f"{base_name}.merged.jsonl"
        with open(merged_file, "w") as out:
            for line in all_lines:
                out.write(line)

        output_file = str(RESULTS_DIR / f"{base_name}.compiled.jsonl")
        if "jsonschemabench" in base_name:
            eval_jsonschemabench(str(merged_file), output_file)
        else:
            eval_jsonschema(str(merged_file), output_file)

        # Clean up merged file
        merged_file.unlink()


if __name__ == "__main__":
    main()

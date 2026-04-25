"""Analyze LAVE false-negative detection results.

Usage:
    python bench/analyze_fn_results.py results/lave_fn_detection_*.jsonl

Prints a summary table plus per-instance breakdown.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(paths: list[str]) -> None:
    all_rows: list[dict] = []
    for p in paths:
        all_rows.extend(load_jsonl(p))

    if not all_rows:
        print("No data found.")
        return

    total_instances = len(all_rows)
    valid_count     = sum(1 for r in all_rows if r.get("valid"))

    # Aggregate FN event counts across all instances
    total_fn   = 0
    total_tn   = 0
    total_skip = 0
    instance_fn_rates = []

    # Distribution of oracle time and mask count for FN events
    fn_oracle_ms  = []
    fn_mask_counts = []
    tn_oracle_ms  = []

    # Prefix vs search breakdown (all oracle events with new fields)
    bucket_stats: dict[str, dict] = {
        "0-3": {"prefix": [], "search": [], "total": [], "n": 0},
        "4-7": {"prefix": [], "search": [], "total": [], "n": 0},
        "8-12": {"prefix": [], "search": [], "total": [], "n": 0},
        "legacy": {"prefix": [], "search": [], "total": [], "n": 0},
    }

    def _collect_timing(e: dict) -> None:
        b = e.get("mask_bucket") or "legacy"
        if b not in bucket_stats:
            b = "legacy"
        st = bucket_stats[b]
        st["n"] += 1
        if e.get("oracle_prefix_sync_ms") is not None:
            st["prefix"].append(float(e["oracle_prefix_sync_ms"]))
        if e.get("oracle_search_ms") is not None:
            st["search"].append(float(e["oracle_search_ms"]))
        if e.get("oracle_ms") is not None:
            st["total"].append(float(e["oracle_ms"]))

    for row in all_rows:
        s = row.get("fn_summary", {})
        n_fn   = s.get("false_negatives", 0)
        n_tn   = s.get("true_negatives",  0)
        n_skip = s.get("skipped_blocks",  0)
        total_fn   += n_fn
        total_tn   += n_tn
        total_skip += n_skip
        if (n_fn + n_tn) > 0:
            instance_fn_rates.append(n_fn / (n_fn + n_tn))

        for e in row.get("fn_events", []):
            if e["type"] == "false_negative":
                fn_oracle_ms.append(e.get("oracle_ms", 0))
                fn_mask_counts.append(e.get("n_masks", 0))
                _collect_timing(e)
            elif e["type"] == "true_negative":
                tn_oracle_ms.append(e.get("oracle_ms", 0))
                _collect_timing(e)

    total_rejects = total_fn + total_tn
    global_fn_rate = total_fn / total_rejects if total_rejects > 0 else 0.0

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("=" * 60)
    print("LAVE FALSE-NEGATIVE DETECTION SUMMARY")
    print("=" * 60)
    print(f"Instances analysed          : {total_instances}")
    print(f"Valid outputs               : {valid_count} ({valid_count/total_instances:.1%})")
    print()
    print(f"Total LAVE reject calls     : {total_rejects}")
    print(f"  Oracle: true negatives    : {total_tn}  (LAVE=False, oracle=False — correct reject)")
    print(f"  Oracle: FALSE NEGATIVES   : {total_fn}  (LAVE=False, oracle=True  — MISSED by LAVE)")
    print(f"  Oracle: skipped (>limit)  : {total_skip}")
    print()
    print(f"Global FN rate              : {global_fn_rate:.2%}  ({total_fn}/{total_rejects})")
    if instance_fn_rates:
        print(f"Per-instance FN rate (mean) : {_mean(instance_fn_rates):.2%}")
        print(f"Instances with ≥1 FN event  : {sum(1 for r in all_rows if r.get('fn_summary',{}).get('false_negatives',0)>0)}")
    print()
    if fn_oracle_ms:
        print(f"Oracle timing (FN events)   : mean={_mean(fn_oracle_ms):.1f}ms  max={max(fn_oracle_ms):.1f}ms")
    if tn_oracle_ms:
        print(f"Oracle timing (TN events)   : mean={_mean(tn_oracle_ms):.1f}ms  max={max(tn_oracle_ms):.1f}ms")
    if fn_mask_counts:
        print(f"MASKs per FN block          : mean={_mean(fn_mask_counts):.1f}  max={max(fn_mask_counts)}")

    # Oracle timing: prefix sync vs block search (priority 1 vs 2 diagnosis)
    has_split = any(
        bucket_stats[k]["prefix"] or bucket_stats[k]["search"]
        for k in ("0-3", "4-7", "8-12")
    )
    if has_split:
        print()
        print("Oracle timing by mask_bucket (FN+TN events, ms)")
        print("(Requires oracle_prefix_sync_ms in JSONL — from oracle_fast runs.)")
        print("-" * 60)
        for label in ("0-3", "4-7", "8-12", "legacy"):
            st = bucket_stats[label]
            n = st["n"]
            if n == 0 and label != "legacy":
                continue
            if n == 0:
                continue
            pm = _mean(st["prefix"]) if st["prefix"] else None
            sm = _mean(st["search"]) if st["search"] else None
            tm = _mean(st["total"]) if st["total"] else None
            ps = f"prefix_sync mean={pm:.1f}" if pm is not None else "prefix_sync n/a"
            ss = f"search mean={sm:.1f}" if sm is not None else "search n/a"
            ts = f"total mean={tm:.1f}" if tm is not None else "total n/a"
            print(f"  {label:<8} n={n:<5}  {ps}  |  {ss}  |  {ts}")
        hi = bucket_stats["8-12"]
        if hi["n"] and hi["prefix"] and hi["search"]:
            pr = _mean(hi["prefix"])
            sr = _mean(hi["search"])
            print()
            print(
                f"  (8-12 masks) prefix_sync / search ratio ≈ "
                f"{pr / sr:.2f}x" if sr > 0 else "  (8-12 masks) search mean is 0"
            )
            print(
                "    → If prefix_sync >> search, bottleneck is incremental replay; "
                "if search dominates, consider DFS cost or bfs_dedup experiments."
            )

    # Per-instance table (instances that had at least one reject)
    noisy = [r for r in all_rows if r.get("fn_summary", {}).get("total_rejects", 0) > 0]
    if noisy:
        print()
        print(f"{'Instance':<40} {'rejects':>7} {'FNs':>5} {'FN%':>6} {'valid':>6} {'retries':>8}")
        print("-" * 80)
        for r in sorted(noisy, key=lambda x: -x.get("fn_summary", {}).get("false_negatives", 0)):
            s    = r.get("fn_summary", {})
            n_fn = s.get("false_negatives", 0)
            tot  = s.get("total_rejects", 0)
            rate = f"{n_fn/tot:.0%}" if tot > 0 else "n/a"
            print(
                f"{r['instance_id']:<40} {tot:>7} {n_fn:>5} {rate:>6} "
                f"{'✓' if r.get('valid') else '✗':>6} {r.get('resamples',0):>8}"
            )


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(
        Path("results").glob("lave_fn_detection_*.jsonl")
    )
    if not files:
        print("Usage: python bench/analyze_fn_results.py results/lave_fn_detection_*.jsonl")
        sys.exit(1)
    analyze([str(f) for f in files])

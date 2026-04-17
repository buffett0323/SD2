"""
Compare LAVE, dgrammar (v2_async_ac4), and DP results across offsets.

Files:
  LAVE:     lave_timed_jsb_medium_s0_t128{_offXXX}.jsonl
  dgrammar: v2_async_ac4_timed_jsb_medium_s0_t128{_offXXX}.jsonl
  DP:       dp_jsb_medium_s0_t128{_offXXX}.jsonl
"""

import json
import os
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

OFFSETS = ["", "_off68", "_off136", "_off204"]

FILE_PATTERNS = {
    "LAVE":    "lave_timed_jsb_medium_s0_t128{offset}.jsonl",
    "dgrammar": "v2_async_ac4_timed_jsb_medium_s0_t128{offset}.jsonl",
    "DP":      "dp_jsb_medium_s0_t128{offset}.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(records: list[dict]) -> dict:
    if not records:
        return {}

    n = len(records)
    valid_count = sum(1 for r in records if r.get("valid"))
    times = [r["time_taken"] for r in records if "time_taken" in r]
    resamples = [r["resamples"] for r in records if "resamples" in r]

    constraint_pcts = [r["timing"]["constraint_pct"] for r in records if "timing" in r and "constraint_pct" in r["timing"]]
    eff_constraint_pcts = [r["timing"]["effective_constraint_pct"] for r in records if "timing" in r and "effective_constraint_pct" in r["timing"]]
    per_tok_total = [r["timing"]["per_token_total_ms"] for r in records if "timing" in r and "per_token_total_ms" in r["timing"]]
    per_tok_constraint = [r["timing"]["per_token_constraint_ms"] for r in records if "timing" in r and "per_token_constraint_ms" in r["timing"]]
    forward_counts = [r["timing"]["forward_count"] for r in records if "timing" in r and "forward_count" in r["timing"]]

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    return {
        "n": n,
        "valid_rate": valid_count / n,
        "valid_count": valid_count,
        "avg_time_s": avg(times),
        "avg_resamples": avg(resamples),
        "avg_constraint_pct": avg(constraint_pcts),
        "avg_eff_constraint_pct": avg(eff_constraint_pcts),
        "avg_per_tok_total_ms": avg(per_tok_total),
        "avg_per_tok_constraint_ms": avg(per_tok_constraint),
        "avg_forward_count": avg(forward_counts),
    }


def print_table(all_stats: dict):
    methods = list(FILE_PATTERNS.keys())
    offset_labels = ["base (0)", "off68", "off136", "off204"]

    metrics = [
        ("n", "N samples", "{:.0f}"),
        ("valid_rate", "Validity (%)", "{:.1%}"),
        ("avg_time_s", "Avg time (s)", "{:.2f}"),
        ("avg_resamples", "Avg resamples", "{:.1f}"),
        ("avg_forward_count", "Avg fwd passes", "{:.1f}"),
        ("avg_constraint_pct", "Constraint overhead (%)", "{:.1f}"),
        ("avg_eff_constraint_pct", "Eff. constraint overhead (%)", "{:.1f}"),
        ("avg_per_tok_total_ms", "Per-token total (ms)", "{:.2f}"),
        ("avg_per_tok_constraint_ms", "Per-token constraint (ms)", "{:.2f}"),
    ]

    print("\n" + "=" * 100)
    print("COMPARISON: LAVE vs dgrammar vs DP  |  dataset: jsb_medium_s0_t128")
    print("=" * 100)

    col_w = 20

    # ── Per-offset breakdown ──────────────────────────────────────────────────
    for oi, offset in enumerate(OFFSETS):
        label = offset_labels[oi]
        print(f"\n{'─'*100}")
        print(f"  Offset: {label}")
        print(f"{'─'*100}")
        header = f"{'Metric':<32}" + "".join(f"{m:>{col_w}}" for m in methods)
        print(header)
        print("-" * len(header))
        for key, display, fmt in metrics:
            row = f"{display:<32}"
            for method in methods:
                val = all_stats[method][offset].get(key, float("nan"))
                try:
                    row += f"{fmt.format(val):>{col_w}}"
                except (ValueError, TypeError):
                    row += f"{'N/A':>{col_w}}"
            print(row)

    # ── Combined (all offsets pooled) ─────────────────────────────────────────
    print(f"\n{'═'*100}")
    print("  Combined (all 4 offsets pooled)")
    print(f"{'═'*100}")
    header = f"{'Metric':<32}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))

    combined: dict[str, list[dict]] = {m: [] for m in methods}
    for method in methods:
        for offset in OFFSETS:
            combined[method].extend(all_stats[method][offset].get("_records", []))

    combined_stats = {m: compute_stats(recs) for m, recs in combined.items()}

    for key, display, fmt in metrics:
        row = f"{display:<32}"
        for method in methods:
            val = combined_stats[method].get(key, float("nan"))
            try:
                row += f"{fmt.format(val):>{col_w}}"
            except (ValueError, TypeError):
                row += f"{'N/A':>{col_w}}"
        print(row)

    print()


def main():
    all_stats: dict[str, dict[str, dict]] = defaultdict(dict)

    for method, pattern in FILE_PATTERNS.items():
        for offset in OFFSETS:
            fname = pattern.format(offset=offset)
            fpath = RESULTS_DIR / fname
            if not fpath.exists():
                print(f"[WARN] missing: {fpath}")
                all_stats[method][offset] = {}
                continue
            records = load_jsonl(fpath)
            stats = compute_stats(records)
            stats["_records"] = records  # keep for pooling
            all_stats[method][offset] = stats

    print_table(all_stats)

    # ── Per-instance matched comparison ──────────────────────────────────────
    print("=" * 100)
    print("PER-INSTANCE MATCH: valid agreement / disagreement across methods")
    print("=" * 100)

    all_records_by_method: dict[str, dict[str, dict]] = defaultdict(dict)
    for method, pattern in FILE_PATTERNS.items():
        for offset in OFFSETS:
            for rec in all_stats[method][offset].get("_records", []):
                iid = rec["instance_id"]
                all_records_by_method[method][iid] = rec

    methods = list(FILE_PATTERNS.keys())
    all_ids = set()
    for m in methods:
        all_ids |= set(all_records_by_method[m].keys())

    agreement = defaultdict(int)
    for iid in all_ids:
        valids = tuple(
            all_records_by_method[m].get(iid, {}).get("valid", None)
            for m in methods
        )
        agreement[valids] += 1

    print(f"\n{'Pattern (LAVE, dgrammar, DP)':<35} {'Count':>8}  {'%':>6}")
    print("-" * 55)
    for pattern, cnt in sorted(agreement.items(), key=lambda x: -x[1]):
        label = str(pattern)
        pct = cnt / len(all_ids) * 100
        print(f"{label:<35} {cnt:>8}  {pct:>5.1f}%")
    print(f"\nTotal unique instances: {len(all_ids)}")


if __name__ == "__main__":
    main()

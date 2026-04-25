"""
Compare DGrammar vs DP on full jsb_medium dataset (off66 multiples).

Usage:
    python bench/compare_dg_dp.py
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Offsets shared by both methods (multiples of 66)
OFFSETS_66 = [0, 66, 132, 198, 264, 330, 396, 462, 528]

PATTERNS = {
    "DGrammar": "v2_async_ac4_timed_jsb_medium_s0_t128{sfx}.jsonl",
    "DP":       "dp_jsb_medium_s0_t128{sfx}.jsonl",
}


def sfx(offset):
    return f"_off{offset}" if offset > 0 else ""


def load_method(pattern):
    records, missing = [], []
    for off in OFFSETS_66:
        path = RESULTS_DIR / pattern.format(sfx=sfx(off))
        if not path.exists():
            missing.append(off)
            continue
        for line in open(path):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if missing:
        print(f"  [WARN] missing offsets: {missing}")
    return records


def pct(lst, lo, hi=None):
    if hi is None:
        return sum(1 for x in lst if x >= lo) / len(lst)
    return sum(1 for x in lst if lo <= x < hi) / len(lst)


def compute_stats(recs):
    n = len(recs)
    times = sorted(r["time_taken"] for r in recs)
    valid = [r for r in recs if r.get("valid")]
    fwd = [r["timing"]["forward_count"] for r in recs]
    res = [r["timing"]["resample_count"] for r in recs]
    cp  = [r["timing"]["constraint_pct"] for r in recs]
    ecp = [r["timing"].get("effective_constraint_pct", float("nan")) for r in recs]
    tok_t = [r["timing"].get("per_token_total_ms", float("nan")) for r in recs]
    tok_c = [r["timing"].get("per_token_constraint_ms", float("nan")) for r in recs]
    saved = [r["timing"].get("mask_time_saved_ms", 0) for r in recs]
    ac    = [r["timing"].get("autocomplete_steps", 0) for r in recs]
    sel   = [r["timing"].get("token_select_total_ms", 0) for r in recs]

    def avg(lst): return statistics.mean(lst) if lst else float("nan")
    def med(lst): return statistics.median(lst) if lst else float("nan")

    return {
        "n": n,
        "valid_rate": len(valid) / n,
        "mean_t": avg(times),
        "median_t": med(times),
        "p95_t": times[int(n * 0.95)],
        "max_t": max(times),
        "mean_fwd": avg(fwd),
        "fwd_lt100_pct": pct(fwd, 0, 100),
        "fwd_eq128_pct": pct(fwd, 128),
        "mean_res": avg(res),
        "mean_cp": avg(cp),
        "mean_ecp": avg([x for x in ecp if x == x]),
        "mean_tok_t": avg([x for x in tok_t if x == x]),
        "mean_tok_c": avg([x for x in tok_c if x == x]),
        "mean_saved_ms": avg(saved),
        "mean_ac_steps": avg(ac),
        "mean_sel_ms": avg(sel),
        "_records": recs,
    }


def print_summary(all_stats):
    methods = list(PATTERNS.keys())
    col = 18

    metrics = [
        ("n",              "N samples",                   "{:.0f}"),
        ("valid_rate",     "Validity",                    "{:.1%}"),
        ("mean_t",         "Mean time (s)",               "{:.2f}"),
        ("median_t",       "Median time (s)",             "{:.2f}"),
        ("p95_t",          "P95 time (s)",                "{:.2f}"),
        ("max_t",          "Max time (s)",                "{:.2f}"),
        ("mean_fwd",       "Avg fwd passes",              "{:.1f}"),
        ("fwd_lt100_pct",  "fwd < 100 (%)",               "{:.1%}"),
        ("fwd_eq128_pct",  "fwd = 128 (%)",               "{:.1%}"),
        ("mean_res",       "Avg resamples",               "{:.2f}"),
        ("mean_cp",        "Constraint % (mean)",         "{:.1f}"),
        ("mean_ecp",       "Eff. constraint % (mean)",    "{:.1f}"),
        ("mean_tok_t",     "Per-token total (ms)",        "{:.2f}"),
        ("mean_tok_c",     "Per-token constraint (ms)",   "{:.3f}"),
        ("mean_saved_ms",  "Mask time saved (ms)",        "{:.0f}"),
        ("mean_ac_steps",  "Avg autocomplete steps",      "{:.1f}"),
        ("mean_sel_ms",    "Token select total (ms)",     "{:.1f}"),
    ]

    print("\n" + "=" * 70)
    print("  DGrammar vs DP  |  jsb_medium, seed=0, T=128  |  offsets: 66×")
    print("=" * 70)
    header = f"{'Metric':<34}" + "".join(f"{m:>{col}}" for m in methods)
    print(header)
    print("-" * len(header))
    for key, label, fmt in metrics:
        row = f"{label:<34}"
        for m in methods:
            val = all_stats[m].get(key, float("nan"))
            try:
                row += f"{fmt.format(val):>{col}}"
            except (ValueError, TypeError):
                row += f"{'N/A':>{col}}"
        print(row)


def print_per_offset(all_stats_by_offset):
    methods = list(PATTERNS.keys())
    col = 14
    print("\n" + "=" * 70)
    print("  Per-offset breakdown")
    print("=" * 70)
    header = f"{'Offset':<8}" + "".join(
        f"{f'{m} valid':>{col}}{f'{m} time':>{col}}{f'{m} fwd':>{col}}"
        for m in methods
    )
    print(header)
    print("-" * len(header))
    for off in OFFSETS_66:
        row = f"{off:<8}"
        for m in methods:
            s = all_stats_by_offset[m].get(off, {})
            if not s:
                row += f"{'—':>{col}}" * 3
                continue
            row += f"{s['valid_rate']:>{col}.1%}"
            row += f"{s['mean_t']:>{col}.2f}"
            row += f"{s['mean_fwd']:>{col}.1f}"
        print(row)


def print_instance_agreement(all_stats):
    dg_map = {r["instance_id"]: r for r in all_stats["DGrammar"]["_records"]}
    dp_map = {r["instance_id"]: r for r in all_stats["DP"]["_records"]}
    all_ids = set(dg_map) | set(dp_map)

    agreement = Counter()
    for iid in all_ids:
        dg_v = dg_map.get(iid, {}).get("valid", None)
        dp_v = dp_map.get(iid, {}).get("valid", None)
        agreement[(dg_v, dp_v)] += 1

    print("\n" + "=" * 70)
    print("  Per-instance validity agreement")
    print("=" * 70)
    print(f"{'(DGrammar, DP)':<25} {'Count':>8}  {'%':>6}")
    print("-" * 45)
    for pattern, cnt in sorted(agreement.items(), key=lambda x: -x[1]):
        pct_val = cnt / len(all_ids) * 100
        print(f"{str(pattern):<25} {cnt:>8}  {pct_val:>5.1f}%")
    print(f"\nTotal matched instances: {len(all_ids)}")

    # DG wins / DP wins
    dg_wins = sum(v for k, v in agreement.items() if k[0] is True and k[1] is not True)
    dp_wins = sum(v for k, v in agreement.items() if k[1] is True and k[0] is not True)
    both    = agreement.get((True, True), 0)
    neither = agreement.get((False, False), 0)
    print(f"\n  Both valid:       {both:>4} ({both/len(all_ids):.1%})")
    print(f"  DGrammar only:    {dg_wins:>4} ({dg_wins/len(all_ids):.1%})")
    print(f"  DP only:          {dp_wins:>4} ({dp_wins/len(all_ids):.1%})")
    print(f"  Neither valid:    {neither:>4} ({neither/len(all_ids):.1%})")


def main():
    all_stats = {}
    all_stats_by_offset = defaultdict(dict)

    for method, pattern in PATTERNS.items():
        print(f"Loading {method}...")
        recs = load_method(pattern)
        all_stats[method] = compute_stats(recs)

        # Per-offset breakdown
        for off in OFFSETS_66:
            path = RESULTS_DIR / pattern.format(sfx=sfx(off))
            if not path.exists():
                continue
            off_recs = [json.loads(l) for l in open(path) if l.strip()]
            if off_recs:
                all_stats_by_offset[method][off] = compute_stats(off_recs)

    print_summary(all_stats)
    print_per_offset(all_stats_by_offset)
    print_instance_agreement(all_stats)
    print()


if __name__ == "__main__":
    main()

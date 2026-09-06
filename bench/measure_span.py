#!/usr/bin/env python3
"""Where does the repair span end, and does the depth-0 guard ever fire?

Reads the `span` / `span_sites_detail` blocks that generate_dp records per
repair site and aggregates them across instances.  The question it answers:
`find_constraint_end` promises the DP span ends at bracket depth 0, but three
of its four exits return the raw lookahead cap without testing depth.  How
often does the guard actually hold?

Usage:
    python bench/measure_span.py --method prog 'results/..._rerun.jsonl' \
                                 --method full 'results/..._full_logp*.jsonl'
"""
import argparse, glob, json, sys
from collections import Counter


def load(patterns):
    rows, seen = [], set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    iid = r.get("instance_id")
                    if iid in seen:
                        continue
                    seen.add(iid)
                    rows.append(r)
    return rows


def agg(rows):
    sites = [s for r in rows for s in r.get("span_sites_detail", [])]
    n_with = sum(1 for r in rows if r.get("span", {}).get("n_sites", 0) > 0)
    total_sites = sum(r.get("span", {}).get("n_sites", 0) for r in rows)
    return sites, n_with, total_sites


def report(name, rows):
    sites, n_with, total_sites = agg(rows)
    print(f"\n=== {name} ===")
    print(f"instances {len(rows)}   with >=1 repair site {n_with}   "
          f"repair sites (all) {total_sites}   sampled sites {len(sites)}")
    if not sites:
        print("  no sites recorded -- rerun with the instrumented build")
        return

    n = len(sites)
    def pct(c):
        return f"{100.0 * c / n:5.1f}%"

    reasons = Counter(s["reason"] for s in sites)
    print("\n  why the span scan stopped:")
    for k in ("junction", "mask", "dead", "lookahead"):
        note = "  <- depth-0 guard held" if k == "junction" else "  <- guard never tested"
        print(f"    {k:<10} {reasons.get(k,0):5d}  {pct(reasons.get(k,0))}{note}")

    unsafe = [s for s in sites if s["reason"] != "junction" and s["depth_before"] > 0]
    closed = [s for s in sites if s["depth_after"] < s["depth_before"]]
    eos = [s for s in sites if s["eos_in_fixes"]]
    print(f"\n  span ends inside an unclosed bracket : {len(unsafe):5d}  {pct(len(unsafe))}")
    print(f"  DP net-closed brackets (the collapse) : {len(closed):5d}  {pct(len(closed))}")
    print(f"  DP inserted EOS/EOT                   : {len(eos):5d}  {pct(len(eos))}")

    spans = [s["span"] for s in sites]
    print(f"\n  span len   mean {sum(spans)/n:5.2f}   ==1 {pct(sum(1 for x in spans if x == 1))}"
          f"   median {sorted(spans)[n//2]}")
    print(f"  depth at span end (model tokens)  mean {sum(s['depth_before'] for s in sites)/n:.2f}")
    print(f"  depth drop caused by the DP       mean {sum(s['depth_before']-s['depth_after'] for s in sites)/n:.2f}")
    print(f"  positions constraint_end promised but MASK truncated away: "
          f"mean {sum(s['constraint_end']-s['eff_end'] for s in sites)/n:.2f}")

    # The joint claim: unsafe end AND the DP took the exit.
    both = sum(1 for s in sites
               if s["reason"] != "junction" and s["depth_before"] > 0
               and s["depth_after"] < s["depth_before"])
    print(f"\n  unsafe end AND net-closed            : {both:5d}  {pct(both)}"
          "   <- the hypothesised mechanism")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", nargs="+", metavar=("NAME", "GLOB"),
                    required=True)
    a = ap.parse_args()
    for spec in a.method:
        report(spec[0], load(spec[1:]))


if __name__ == "__main__":
    main()

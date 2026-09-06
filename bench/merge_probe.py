#!/usr/bin/env python3
"""Does merging on the state key discard the optimum?

The DP keys lattice nodes on ``bytes(compute_logit_bias())`` -- the tokens the
parser admits next -- and collapses every path reaching a key to the best one.
That key is a state *abstraction*, not a state identifier: two parser states can
admit the same tokens and still differ in what the grammar demands later, so the
merge is sound (every surviving path is one a parser consumed) but not obviously
exact.  This script measures the gap, in two tiers, because the honest oracle
and the paper's operating point cannot both be run at once.

  Tier A -- the real lattice (``--top-k``, the paper's 100).  Beam widths
      1, 2, 4, ... per key.  Widening the beam strictly enlarges the searched
      set, so a width that never changes the answer is evidence the collapse to
      one costs nothing.  Not a proof: no-merge at k=100 is k^span paths.

  Tier B -- a small lattice (``--exact-top-k``, e.g. 6, over
      ``--exact-span`` positions).  Small enough to run with no merging at all,
      which returns the lattice's true optimum.  When the exact arm reports
      ``exhaustive`` (the ``--max-live`` cap never bound), a site where it agrees
      with beam=1 is a *proof* that merging cost nothing there.

Sites are the structural positions of real outputs -- outside string literals and
with a legal set below ``--max-legal`` -- which is where violations occur and
where the parser is doing grammatical rather than lexical work.  Inside a
free-form string every continuation is content and the question is empty.

No model and no GPU: parser states come from replaying generated outputs through
llguidance, the same teacher-forced replay the legal-set measurement uses, and
the model's role -- ranking candidates -- is played by random score vectors.
That is deliberately harder than the real thing; the merge either survives
thousands of independent orderings of the same candidate sets, or it does not.

    python bench/merge_probe.py --instances 20 --seeds 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgrammar.dp_generate import dp_fix_prefix   # noqa: E402

MASK_ID = 126336        # LLaDA-8B-Instruct; never appears in a decoded output
EOS_ID, EOT_ID = 126081, 126348


def in_string(s: str) -> bool:
    """Is the decoded prefix currently inside a JSON string literal?"""
    esc = ins = False
    for c in s:
        if esc:
            esc = False
            continue
        if c == "\\" and ins:
            esc = True
            continue
        if c == '"':
            ins = not ins
    return ins


def load_rows(patterns: list[str]) -> list[dict]:
    """One row per instance id, preferring the longest output for duplicates."""
    best: dict[str, dict] = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not (r.get("schema") and r.get("extracted")):
                        continue
                    iid = r.get("instance_id") or path
                    prev = best.get(iid)
                    if prev is None or len(r["extracted"]) > len(prev["extracted"]):
                        best[iid] = r
    return sorted(best.values(), key=lambda r: str(r.get("instance_id")))


def score_vector(rng, vocab: int, orig_tok: int, temp: float,
                 boost: float) -> np.ndarray:
    """A peaked log-probability vector standing in for the model's."""
    z = rng.standard_normal(vocab).astype(np.float64) * temp
    if 0 <= orig_tok < vocab:
        z[orig_tok] += boost
    z -= z.max()
    return (z - np.log(np.exp(z).sum())).astype(np.float32)


def solve(base, x, p, lp, k, end, beam, max_live):
    out: dict = {}
    _, reached = dp_fix_prefix(
        base.deep_copy(), x, p, lp, MASK_ID, top_k=k, end_pos=end,
        eos_id=EOS_ID, eot_id=EOT_ID,
        beam_per_key=beam, max_live=max_live, out=out)
    return {
        "assignment": out.get("assignment"),
        "score": out.get("best_score"),
        "reached": reached,
        "exhaustive": bool(out.get("exhaustive")),
        "capped": bool(out.get("capped")),
        "n_live_max": out.get("n_live_max"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*",
                    default=["results/dp_jsb_medium_s0_t128*v6dp.jsonl"])
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    # Tier A: the real lattice.
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--span", type=int, default=8)
    ap.add_argument("--beams", default="2,4,8", help="widths compared against 1")
    # Tier B: small enough to enumerate.
    ap.add_argument("--exact-top-k", type=int, default=6)
    ap.add_argument("--exact-span", type=int, default=4)
    ap.add_argument("--max-live", type=int, default=20000)
    # Site selection.
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-legal", type=int, default=4000,
                    help="skip states admitting more than this many tokens")
    ap.add_argument("--max-sites-per-instance", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=3.0)
    ap.add_argument("--orig-boost", type=float, default=2.0)
    ap.add_argument("--site-budget-s", type=float, default=25.0)
    ap.add_argument("--out", default="results/merge_probe.jsonl")
    args = ap.parse_args()

    from dgrammar.checker import TokenChecker

    beams = [int(b) for b in args.beams.split(",") if b.strip()]
    rows = load_rows(args.results)[args.offset:args.offset + args.instances]
    print(f"[merge_probe] {len(rows)} instances | tierA k={args.top_k} span={args.span} "
          f"beams={beams} | tierB k={args.exact_top_k} span={args.exact_span} "
          f"| seeds={args.seeds}", flush=True)

    fh = open(args.out, "w")
    t0 = time.time()
    n_sites = n_trials = 0
    a_diff, a_lost = Counter(), Counter()
    b_trials = b_exhaustive = b_diff = b_lost = 0
    skipped = 0

    for ri, row in enumerate(rows):
        iid = row.get("instance_id")
        try:
            ck = TokenChecker(row["schema"])
        except Exception as exc:
            skipped += 1
            print(f"  [{ri}] {iid}: skip ({type(exc).__name__})", flush=True)
            continue

        toks = list(ck.tokenizer.tokenize_str(row["extracted"].rstrip("\n")))
        if len(toks) < 4:
            continue
        vocab = len(ck.matcher.compute_logit_bias())
        x = torch.tensor([toks], dtype=torch.long)
        lp = torch.zeros((1, len(toks), vocab), dtype=torch.float32)
        rng = np.random.default_rng(1234 + args.offset + ri)

        prefix = ""
        sites = 0
        for p in range(len(toks)):
            take = (p >= 1 and p % args.stride == 0 and p + 2 <= len(toks)
                    and sites < args.max_sites_per_instance
                    and not in_string(prefix))
            if take:
                mask = np.frombuffer(ck.matcher.compute_logit_bias(), dtype=np.uint8)
                n_legal = int(np.count_nonzero(mask))
                if n_legal == 0 or n_legal > args.max_legal:
                    take = False
            if take:
                base = ck.matcher.deep_copy()
                endA = min(p + args.span, len(toks))
                endB = min(p + args.exact_span, len(toks))
                t_site = time.time()
                for s in range(args.seeds):
                    for q in range(p, endA):
                        lp[0, q] = torch.from_numpy(
                            score_vector(rng, vocab, toks[q], args.temp, args.orig_boost))
                    try:
                        a1 = solve(base, x, p, lp, args.top_k, endA, 1, args.max_live)
                        aN = {f"B{b}": solve(base, x, p, lp, args.top_k, endA, b,
                                             args.max_live) for b in beams}
                        b1 = solve(base, x, p, lp, args.exact_top_k, endB, 1, args.max_live)
                        bx = solve(base, x, p, lp, args.exact_top_k, endB, None,
                                   args.max_live)
                    except Exception as exc:
                        print(f"    p={p}: {type(exc).__name__}: {exc}", flush=True)
                        break

                    n_trials += 1
                    rec = {"instance_id": iid, "pos": p, "seed": s,
                           "n_legal": n_legal,
                           "tierA": {"span": endA - p, "k": args.top_k,
                                     "b1_score": a1["score"],
                                     "b1_live_max": a1["n_live_max"], "arms": {}},
                           "tierB": {"span": endB - p, "k": args.exact_top_k,
                                     "b1_score": b1["score"],
                                     "b1_live_max": b1["n_live_max"],
                                     "exact_score": bx["score"],
                                     "exhaustive": bx["exhaustive"],
                                     "n_live_max": bx["n_live_max"]}}
                    for name, r in aN.items():
                        d = r["assignment"] != a1["assignment"]
                        l = (r["score"] is not None and a1["score"] is not None
                             and r["score"] > a1["score"] + 1e-9)
                        a_diff[name] += d
                        a_lost[name] += l
                        rec["tierA"]["arms"][name] = {"differs": d, "lost": l,
                                                      "score": r["score"]}
                    b_trials += 1
                    if bx["exhaustive"]:
                        b_exhaustive += 1
                        d = bx["assignment"] != b1["assignment"]
                        l = (bx["score"] is not None and b1["score"] is not None
                             and bx["score"] > b1["score"] + 1e-9)
                        b_diff += d
                        b_lost += l
                        rec["tierB"]["differs"] = d
                        rec["tierB"]["lost"] = l
                    fh.write(json.dumps(rec) + "\n")
                    if time.time() - t_site > args.site_budget_s:
                        break
                n_sites += 1
                sites += 1

            if ck.matcher.try_consume_tokens([toks[p]]) != 1:
                break
            prefix += ck.tokenizer.decode_str([toks[p]])

        fh.flush()
        print(f"  [{ri}] {iid}: {sites} sites, {n_trials} trials, "
              f"{time.time()-t0:.0f}s", flush=True)

    fh.close()
    pct = lambda a, b: f"{100.0*a/max(b,1):.2f}%"
    print("\n" + "=" * 70)
    print(f"instances {len(rows)-skipped} (skipped {skipped}) | sites {n_sites} | trials {n_trials}")
    print(f"\nTier A -- real lattice (k={args.top_k}, span<={args.span}), vs beam=1:")
    for name in (f"B{b}" for b in beams):
        print(f"  {name:>4}: assignment differs {a_diff[name]}/{n_trials} ({pct(a_diff[name], n_trials)})"
              f" | beam=1 strictly worse {a_lost[name]}/{n_trials} ({pct(a_lost[name], n_trials)})")
    print(f"\nTier B -- enumerable lattice (k={args.exact_top_k}, span<={args.exact_span}):")
    print(f"  exhaustive on {b_exhaustive}/{b_trials} trials ({pct(b_exhaustive, b_trials)})")
    print(f"  vs the true optimum: differs {b_diff}/{b_exhaustive} ({pct(b_diff, b_exhaustive)})"
          f" | beam=1 strictly worse {b_lost}/{b_exhaustive} ({pct(b_lost, b_exhaustive)})")
    print(f"\nwritten to {args.out}")
    print("(a Tier-B trial whose exact arm carried many more live paths than "
          "beam=1 is one where merging actually did work)")


if __name__ == "__main__":
    main()

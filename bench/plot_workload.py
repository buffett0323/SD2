#!/usr/bin/env python3
"""Figure: where DPGrammar's extra wall time goes, on the instances it fires.

A time breakdown would be dishonest here: the instrumented counters cover only
about a quarter of wall clock in both arms, so a stacked bar would be mostly
"unattributed".  What *is* fully measured is the work done, and it explains the
gap without appealing to the residual: the layer emits far more tokens, which
needs proportionally more forward passes, on a longer sequence that makes each
forward slower.  The constraint machinery itself costs the same in both arms.
"""
import json, glob, sys, statistics
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "bench"); sys.path.insert(0, ".")
from measure_valid import is_infra_failure

def load(pat):
    r = {}
    for f in glob.glob(pat):
        for ln in open(f):
            if ln.strip():
                d = json.loads(ln); r.setdefault(d["instance_id"], d)
    return r

D = load("results/*v6dp.jsonl"); B = load("results/*v6base.jsonl")
keys = [k for k, r in D.items() if not is_infra_failure(r) and k in B]
fired = [k for k in keys if (D[k].get("span_sites_detail") or [])]
quiet = [k for k in keys if not (D[k].get("span_sites_detail") or [])]

def mean(a, f):
    return statistics.mean([float((a[k].get("timing") or {}).get(f) or 0) for k in fired])
def wall(a, ks):
    return statistics.mean([float(a[k]["time_taken"]) for k in ks])

BASE, OURS = "#7f8fa4", "#2f6f4f"
fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.35),
                       gridspec_kw={"width_ratios": [1.15, 1, 1]})

# (a) where the cost falls
w = 0.34; x = np.arange(2)
b = [wall(B, quiet), wall(B, fired)]
d = [wall(D, quiet), wall(D, fired)]
ax[0].bar(x - w/2, b, w, color=BASE, label="no-DP")
ax[0].bar(x + w/2, d, w, color=OURS, label="DPGrammar")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"DP never fires\n({len(quiet)})",
                                            f"DP fires\n({len(fired)})"])
ax[0].set_ylabel("wall clock (s)")
ax[0].set_title("(a) the cost is not spread evenly", fontsize=9, loc="left")
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
for i, (bb, dd) in enumerate(zip(b, d)):
    ax[0].annotate(f"{dd-bb:+.1f}s", (i, max(bb, dd)), textcoords="offset points",
                   xytext=(0, 3), ha="center", fontsize=7.5, color="0.25")

# (b) what the extra time is spent producing
labels = ["output\ntokens", "forward\npasses", "remasks"]
bv = [mean(B, "tokens_unmasked"), mean(B, "forward_count"), mean(B, "resample_count")]
dv = [mean(D, "tokens_unmasked"), mean(D, "forward_count"), mean(D, "resample_count")]
x = np.arange(3)
ax[1].bar(x - w/2, bv, w, color=BASE); ax[1].bar(x + w/2, dv, w, color=OURS)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7.5)
ax[1].set_ylabel("per instance")
ax[1].set_title("(b) it does more of the same work", fontsize=9, loc="left")

# (c) unit costs
labels = ["ms per\noutput token", "ms per\nforward", "constraint\nms (total)"]
bu = [wall(B, fired)*1000/bv[0], mean(B,"forward_total_ms")/bv[1],
      mean(B,"mask_compute_total_ms")+mean(B,"grammar_check_total_ms")+mean(B,"token_select_total_ms")]
du = [wall(D, fired)*1000/dv[0], mean(D,"forward_total_ms")/dv[1],
      mean(D,"mask_compute_total_ms")+mean(D,"grammar_check_total_ms")+mean(D,"token_select_total_ms")]
x = np.arange(3)
ax[2].bar(x - w/2, bu, w, color=BASE); ax[2].bar(x + w/2, du, w, color=OURS)
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=7.5)
ax[2].set_yscale("log"); ax[2].set_ylabel("ms (log)")
ax[2].set_title("(c) each unit costs about the same", fontsize=9, loc="left")

for a in ax:
    for side in ("top", "right"): a.spines[side].set_visible(False)
    a.tick_params(labelsize=7.5)
    a.yaxis.label.set_size(8)
fig.tight_layout(pad=0.4)
fig.savefig("neurips/workload.pdf")
print(f"fired={len(fired)} quiet={len(quiet)}")
print(f"  tokens {bv[0]:.1f} -> {dv[0]:.1f} | forwards {bv[1]:.1f} -> {dv[1]:.1f} | remasks {bv[2]:.1f} -> {dv[2]:.1f}")
print(f"  ms/token {bu[0]:.1f} -> {du[0]:.1f} | ms/forward {bu[1]:.1f} -> {du[1]:.1f} | constraint {bu[2]:.0f} -> {du[2]:.0f}")
print("wrote neurips/workload.pdf")

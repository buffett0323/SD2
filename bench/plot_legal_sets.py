#!/usr/bin/env python3
"""Figure: the distribution of |L(q)| over live parser states.

A percentile table hides what matters here.  The distribution is bimodal, and
the gap between the modes spans three orders of magnitude, so the reader has to
infer it from p50=106 against p75=125,307.  On a log axis the two modes and the
empty decade between them are visible at a glance.

Data: teacher-forced replay of 40 DPGrammar outputs on JSONSchemaBench medium
through llguidance 1.7.0, recording the mask size before each consumed token.
Regenerates from results/ if the cached sizes are absent.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "/tmp/lq_sizes.json"
VOCAB = 126_349

def in_string(s: str) -> bool:
    esc = ins = False
    for c in s:
        if esc: esc = False; continue
        if c == "\\" and ins: esc = True; continue
        if c == '"': ins = not ins
    return ins

def measure():
    sys.path.insert(0, "."); sys.path.insert(0, "bench")
    from measure_valid import load
    from dgrammar.checker import TokenChecker
    D = load(["results/dp_jsb_medium_s0_t128_v6dp.jsonl"], "")
    rows = [r for r in D.values() if r.get("schema") and r.get("extracted")]
    struct, instr = [], []
    for r in rows[:40]:
        try: ck = TokenChecker(r["schema"])
        except Exception: continue
        pref = ""
        for t in ck.tokenizer.tokenize_str(r["extracted"].rstrip("\n")):
            n = int(np.count_nonzero(
                np.frombuffer(ck.matcher.compute_logit_bias(), dtype=np.uint8)))
            (instr if in_string(pref) else struct).append(n)
            if ck.matcher.try_consume_tokens([t]) != 1: break
            pref += ck.tokenizer.decode_str([t])
    return struct, instr

if os.path.exists(CACHE) and "--fresh" not in sys.argv:
    d = json.load(open(CACHE))
    struct, instr = (d["structural"], d["in_string"]) if isinstance(d, dict) else (d, [])
else:
    struct, instr = measure()
    json.dump({"structural": struct, "in_string": instr}, open(CACHE, "w"))

if not instr:                      # cache predates the split; recompute
    struct, instr = measure()
    json.dump({"structural": struct, "in_string": instr}, open(CACHE, "w"))

bins = np.logspace(0, np.log10(VOCAB), 45)
fig, ax = plt.subplots(figsize=(5.4, 2.5))
ax.hist([struct, instr], bins=bins, stacked=True, linewidth=0.4,
        edgecolor="white",
        color=["#3b6ea5", "#c96f3f"],
        label=[f"structural position (n = {len(struct):,})",
               f"inside a string (n = {len(instr):,})"])
ax.set_xscale("log")
ax.set_xlabel("tokens the parser can consume at a live state  (log scale)")
ax.set_ylabel("live states")

# Headroom so the legend clears the tallest bar; without it the legend text
# lands on the structural mode at ~10^2.
ymax = max(np.histogram(struct + instr, bins=bins)[0])
ax.set_ylim(0, ymax * 1.42)

ax.axvline(VOCAB, color="0.45", ls=":", lw=0.9)
ax.annotate("vocabulary", xy=(VOCAB, ymax * 1.30), xytext=(-3, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=8, color="0.45")
ax.legend(frameon=False, fontsize=8, loc="upper left",
          bbox_to_anchor=(0.02, 1.0), handlelength=1.2, handleheight=0.9)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.tick_params(labelsize=8)
ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
fig.tight_layout(pad=0.3)
fig.savefig("neurips/legal_sets.pdf")
tot = len(struct) + len(instr)
allv = np.array(struct + instr)
print(f"states={tot}  <100: {100*(allv<100).mean():.1f}%  "
      f">10k: {100*(allv>10_000).mean():.1f}%  median={int(np.median(allv))}")
print("wrote neurips/legal_sets.pdf")

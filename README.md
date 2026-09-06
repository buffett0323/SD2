# DPGrammar — Joint Viterbi Repair for Grammar-Constrained Diffusion Language Models

Jeng-Yue Liu, Wilson Zheng, Haoling Pu
Language Technologies Institute, Carnegie Mellon University

Paper: [`neurips/main.pdf`](neurips/main.pdf) · source in [`neurips/`](neurips/)

---

## What this is

A masked diffusion LM reveals many positions per step, so by the time an
incremental parser rejects a token, the tokens around it are already committed.
The usual answer is to hand the position back to the sampler and let it try
again, one token at a time. That works when a violation is local and fails when
it is not: the cheapest legal token at a violation is often a closing bracket,
so the document terminates early instead of being repaired.

**DPGrammar** decides the violated span as a whole. It runs a Viterbi pass over
(parser state × position) pairs, merging paths that leave the parser in the same
state, at a cost linear in the span length rather than exponential in it.
Candidates at each node come from the parser automaton's outgoing edges rather
than from the model's top-*k*, so every candidate is one the parser will accept
and a tighter budget costs likelihood, never validity.

The layer is schedule-agnostic: it fires only after a parser rejection, so it
places no requirement on the order in which positions are revealed.

## Results

JSONSchemaBench `medium`, LLaDA-8B-Instruct, seed 0, T=128, on the 511 of 586
schemas `llguidance` compiles:

| | schema@1 | content (≥5 leaves) | hand-backs/inst | mean s |
|---|---|---|---|---|
| Vanilla | 49.8 | 47.1 | — | 14.89 |
| CD-CFG | 64.8 | 58.0 | 57.3 | 22.71 |
| LAVE | 78.3 | 74.0 | 253.5 | 38.50 |
| no-DP (ours) | 91.8 | 74.6 | 33.2 | **13.45** |
| **DPGrammar** | **96.7** | **84.9** | **1.2** | 16.75 |

Validity rises 4.9 pp over the unrepaired baseline while hand-backs fall 28×.
The margin more than doubles, to 10.3 pp, once an output must carry five
populated leaves: single-token repair buys validity by terminating documents
early, and the content floor scores that as the failure it is. Full tables,
including `easy`/`hard`/JSON-Mode-Eval and the ablations, are in the paper.

## Layout

```
dgrammar/            the method
  checker.py           incremental parser wrapper (llguidance 1.7.0)
  dp_generate.py       dp_fix_prefix: the Viterbi pass, and the DP decode loop
  generate.py          single-token-retry decode loop (the no-DP baseline)
  fa_generate.py       exact inference over a finite-automaton posterior
                       (the prevention arm of Appendix G)

bench/
  run_dgrammar_timed.py    DPGrammar and no-DP runners
  run_lave_timed.py        LAVE
  run_igcd_timed.py        CD-CFG, and Vanilla when --constrain False
  run_fa_timed.py          prevention arm
  merge_probe.py           what the state-merge key costs (Appendix F)
  modal_*.py               Modal wrappers for each of the above
  measure_*.py             offline measurements over saved runs
  plot_legal_sets.py       Figure 2
  plot_workload.py         Figure 3
  audit_paper_numbers.py   recomputes Table 1 from results/ and diffs it
                           against the values printed in the paper

results/     the runs every reported number is computed from
neurips/     the paper
docs/        review_response.md tracks the review rounds; the rest is the
             original course submission (report, poster, proposal)
not_used/    superseded runs and scripts, git-ignored (see below)
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv pip install -e .
uv pip install 'llguidance>=1.7' jsonschema datasets
export HF_TOKEN=...        # LLaDA-8B-Instruct and the benchmarks are gated
```

The runners import `constrained_diffusion.eval.dllm.*` for dataset and model
loading. Only the dataset half is vendored here; the rest comes from the
upstream checkouts under `vendor/` (git-ignored):

```bash
git clone https://github.com/eth-sri/constrained-diffusion vendor/constrained-diffusion
git clone <CD4dLLM>                                        vendor/CD4dLLM
cd vendor/constrained-diffusion/rustformlang_bindings && maturin develop --release
```

`vendor/patch_cd4dllm.sh` applies the changes CD-CFG needs to run headless.

## Reproducing the reported runs

Every reported file is `<method>_<dataset>_s0_t128[_offN][_<tag>].jsonl` under
`results/`. Shards are merged on `instance_id`. Each command below writes the
tag the paper's numbers are read from, so re-running one overwrites exactly the
files the audit script reads.

```bash
# DPGrammar / no-DP, per split.  --method dp -> dp_*, --method dgrammar -> v2_async_ac4_timed_*
modal run bench/modal_dgrammar_bench.py --dataset jsb_medium --total 586 --chunks 9 \
    --method dp        --block-ar 1 --tag v6dp
modal run bench/modal_dgrammar_bench.py --dataset jsb_medium --total 586 --chunks 9 \
    --method dgrammar  --block-ar 1 --tag v6base

# LAVE and CD-CFG / Vanilla
modal run bench/modal_lave_bench.py --dataset jsb_medium --total 586 --chunks 9 --tag v6lave
modal run bench/modal_igcd_bench.py --dataset jsb_medium --total 586 --chunks 9 --constrain True
modal run bench/modal_igcd_bench.py --dataset jsb_medium --total 586 --chunks 9 --constrain False
```

Tags for the other splits: `easy` (`--total 577`) uses `easydp`/`easybase`/`easylave`,
`hard` (`--total 368`) uses `harddp3`/`hardbase3`/`hardlave`, and JSON-Mode-Eval
(`--dataset jsonschema --total 272`) uses `jmdp`/`jmbase`/`jmlave`. CD-CFG and
Vanilla take no tag.

Ablations and the schedule study vary one flag and change the tag:

```bash
--cand-source vocab       --tag abl_vocab      # Table 4, candidate source
--objective minedit       --tag abl_minedit    # Table 4, objective
--eos-in-candidates True  --tag abl_eos        # Table 4, stop tokens
--remasking random        --tag randorder      # Table 8, random unmasking
--block-ar 0              --tag fullpar        # Table 8, full parallel
```

The merge probe and the prevention arm are separate apps:

```bash
modal run bench/modal_merge_probe.py --instances 512 --chunks 16 --beams "1,2,4" --tag full
modal run bench/modal_fa_bench.py --task jsb_medium --decoder viterbi  --total 128 --tag full
modal run bench/modal_fa_bench.py --task jsb_medium --decoder marginal --total 128 --tag full
```

`merge_probe.py` needs no GPU: it replays saved outputs through the parser and
scores the lattice with random vectors, so the merge has to survive many
orderings of the same candidate sets rather than one. The candidate-budget
comparison of Appendix A is the same replay at two budgets:

```bash
python bench/merge_probe.py --top-k 100 --out results/merge_probe_k100.jsonl
python bench/merge_probe.py --top-k 128 --out results/merge_probe_k128.jsonl
```

## How the numbers are computed

Three conventions matter, and getting any of them wrong moves the tables:

1. **Validation uses `jsonschema.validate`**, which picks the validator from the
   instance's own `$schema`. Pinning `Draft7Validator` instead shifts `medium`
   by one to two instances per arm.
2. **The denominator is the set of schemas the arm's toolchain compiles** — 558 /
   511 / 269 for `llguidance`, 449 / 383 / 245 for CD-CFG's `rustformlang`. A row
   whose grammar did not compile has `time_taken` under `0.01`; it counts as
   invalid but is excluded from the latency statistics. Vanilla is scored on the
   full split, since it has no grammar.
3. **Percentiles**: p50 is the median, p95 and p99 are nearest-rank.
4. **CD-CFG is scored on its `autocompletion` field**, falling back to
   `extracted` where the released code produced none. That is the configuration
   its authors ship, and it is worth about 20 pp against scoring `extracted`.

```bash
python bench/audit_paper_numbers.py nearestrank     # 0 mismatches against the paper
```

## not_used/

Everything under `not_used/` was superseded and is git-ignored. It holds the
earlier ACL write-up, LAVE variants and oracle experiments that no reported
number depends on, and 158 result files from runs that later re-ran under a new
tag (`harddp`/`harddp2` before `harddp3`, the `gl512` and `rerun` sharding, the
`spanbase`/`spanfull` window study). Nothing in the paper reads from it; delete
the directory if you want the disk back.

## References

LLaDA (Nie et al., 2025) · llguidance (Microsoft, 2025) · JSONSchemaBench
(Geng et al., 2025) · LAVE (Zhang et al., 2026) · CD4dLLM (Cai et al., 2026) ·
constrained discrete diffusion (Cardei et al., 2025)

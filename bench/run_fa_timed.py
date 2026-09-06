"""Benchmark runner for proactively aligned FA decoding (``dgrammar.fa_generate``).

Kept separate from ``run_dgrammar_timed.py`` on purpose: most of that file --
autocomplete passes, force-close, minimal-JSON synthesis, the resample budget,
the instance wall-clock budget for repair -- exists to clean up violations.
Under proactive alignment violations cannot occur, so all of it is dead code
here, and carrying it would obscure what the method actually costs.

Emits the same JSONL schema as ``run_dgrammar_timed.py`` so
``bench/measure_degeneracy.py`` and the existing eval scripts work unchanged.

The 2x2 the experiments need:

    reactive  x mode    run_dgrammar_timed.py --method dp        (current DPGrammar)
    reactive  x mass    run_dgrammar_timed.py --method dp_soft   (not yet built)
    proactive x mode    run_fa_timed.py --decoder viterbi
    proactive x mass    run_fa_timed.py --decoder marginal       (proposed method)

Usage
-----
    python bench/run_fa_timed.py --task json       --decoder marginal --limit 272
    python bench/run_fa_timed.py --task smiles     --decoder marginal --max-depth 3
    python bench/run_fa_timed.py --task jsb_medium --decoder marginal

Per-benchmark differences (regex source, code fence, validity checker) live in
``bench/fa_tasks.py``; CPP is still blocked there on the real C++ grammar.
Validity is judged by an external checker, never by the automaton that produced
the string -- appendix C of the report makes that point: EOS validity is not
schema validity.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from dgrammar.fa_generate import FAStats, TokenDFA, finalize, generate_fa
from fa_tasks import TASKS

import jsb_dataset  # noqa: F401 - registers jsb_* datasets

# LLaDA-8B-Instruct special tokens.
MASK_ID, EOS_ID, EOT_ID = 126336, 126081, 126348
VOCAB_SIZE = 126464


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--decoder", default="marginal",
                    choices=["marginal", "viterbi", "sample"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="only affects --decoder sample")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="branch-nesting bound for non-regular grammars. "
                         "SMILES needs 3 for 100%% reference coverage; C++ "
                         "reference bodies reach 5.")
    ap.add_argument("--max-edges", type=int, default=20_000_000,
                    help="skip instances whose lifted automaton exceeds this; "
                         "per-step cost is O(L * E)")
    ap.add_argument("--instance-ids", default="",
                    help="comma-separated ids; overrides --limit/--offset")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    spec = TASKS[args.task]
    method_tag = f"fa_{args.decoder}"
    sfx = f"_off{args.offset}" if args.offset > 0 else ""
    tag_sfx = f"_{args.tag}" if args.tag else ""
    output_file = (
        f"results/{method_tag}_{args.task}_s{args.seed}_t{args.steps}{sfx}{tag_sfx}.jsonl"
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print(f"{spec.name} | FA [{args.decoder}] | "
          f"{'regular grammar' if spec.regular else f'depth-bounded to {args.max_depth}'}")
    if not spec.regular:
        print("  note: this grammar is not regular; a DFA only exists under the "
              "nesting bound, so coverage is reported below.")

    dataset = load_dataset(spec.dataset)
    eval_model = load_model(args.model)
    torch.manual_seed(args.seed)
    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    if args.instance_ids:
        wanted = set(args.instance_ids.split(","))
        instances = [i for i in all_instances if i.instance_id() in wanted]
    else:
        instances = all_instances[args.offset: args.offset + args.limit]

    print(f"  {len(instances)} instances, seed={args.seed}, T={args.steps}, "
          f"gen_length={args.gen_length} -> {output_file}")

    # Vocabulary is loaded once and reused; Index construction is per grammar.
    from outlines_core import Index, Vocabulary

    t0 = time.perf_counter()
    voc = Vocabulary.from_pretrained(args.model)
    print(f"  vocabulary loaded in {time.perf_counter() - t0:.1f}s "
          f"(eos={voc.get_eos_token_id()})")

    dfa_cache: dict[str, tuple[TokenDFA, float]] = {}
    skips: collections.Counter = collections.Counter()
    done = 0
    agg_expected: list[float] = []
    agg_decoded: list[float] = []

    def write(rec: dict) -> None:
        with open(output_file, "a") as fh:
            print(json.dumps(rec), flush=True, file=fh)

    def skip(instance, reason: str) -> None:
        skips[reason.split(":")[0]] += 1
        rec = {
            "instance_id": instance.instance_id(), "method": method_tag,
            "task": args.task, "valid": False, "extracted": None,
            "time_taken": None, "resamples": 0,
            "timed_out": True, "timed_out_reason": reason,
        }
        data = getattr(instance, "data", None)
        if isinstance(data, dict) and data.get("schema") is not None:
            rec["schema"] = data["schema"]
        write(rec)

    for i, instance in enumerate(instances):
        iid = instance.instance_id()

        # ── constraint ────────────────────────────────────────────────────
        # JSON and C++ carry a distinct grammar per instance so the automaton
        # cannot be amortised; SMILES shares one across the whole benchmark.
        try:
            schema = (getattr(instance, "data", {}) or {}).get("schema")
            key = iid if spec.per_instance_grammar else spec.name
            if key not in dfa_cache:
                dfa_cache.clear()
                tb = time.perf_counter()
                rx = spec.build_regex(instance, args.max_depth)
                dfa = TokenDFA.from_outlines_index(
                    Index(rx, voc), vocab_size=VOCAB_SIZE, eos_id=EOS_ID
                )
                dfa_cache[key] = (dfa, time.perf_counter() - tb)
            dfa, build_s = dfa_cache[key]
        except NotImplementedError as e:
            print(f"  [{i+1}/{len(instances)}] {iid}: {e}")
            skip(instance, f"dfa_not_implemented: {e}")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(instances)}] {iid}: dfa_build_error: {e}")
            skip(instance, f"dfa_build_error: {e}")
            continue

        if dfa.num_edges > args.max_edges:
            print(f"  [{i+1}/{len(instances)}] {iid}: automaton too large "
                  f"(E={dfa.num_edges:,})")
            skip(instance, f"automaton_too_large: E={dfa.num_edges}")
            continue

        # ── prompt ────────────────────────────────────────────────────────
        try:
            prompt_ids, prompt_len, *_ = eval_model.prepare_prompt(
                instance, tokenizer, model, trace=False
            )
        except Exception as e:  # noqa: BLE001
            skip(instance, f"prompt_error: {e}")
            continue

        # ── generate ──────────────────────────────────────────────────────
        print(f"[{i+1}/{len(instances)}] {iid}  S={dfa.num_states} "
              f"E={dfa.num_edges:,} build={build_s:.1f}s ...", flush=True)

        stats = FAStats(build_time=build_s)
        torch.manual_seed(args.seed)
        start = time.monotonic()
        out = None
        try:
            for out, _done in generate_fa(
                model, prompt_ids, dfa,
                steps=args.steps, gen_length=args.gen_length,
                decoder=args.decoder, temperature=args.temperature,
                mask_id=MASK_ID, eos_id=EOS_ID, seed=args.seed, stats=stats,
            ):
                pass
        except Exception as e:  # noqa: BLE001
            print(f"    decode_error: {e}")
            skip(instance, f"decode_error: {e}")
            continue
        elapsed = time.monotonic() - start

        raw = finalize(out, prompt_ids.shape[1], EOS_ID, tokenizer)
        extracted = raw
        try:
            got = instance.extract_result(raw)
            if got and got.strip():
                extracted = got
        except Exception:  # noqa: BLE001
            pass

        valid = spec.valid(extracted, instance)
        passed = spec.functional(extracted, instance)

        timing = stats.summary()
        result = {
            "instance_id": iid,
            "method": method_tag,
            "task": args.task,
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": 0,          # structurally zero: no repair path exists
            "timing": {
                **timing,
                # named to match run_dgrammar_timed.py so comparisons line up
                "total_forward_ms": timing["forward_total_s"] * 1000,
                "total_constraint_ms": timing["infer_total_s"] * 1000,
                "constraint_pct": timing["constraint_overhead_pct"],
                "dfa_build_ms": build_s * 1000,
            },
        }
        if passed is not None:
            result["passed_tests"] = passed
        if schema:
            result["schema"] = schema

        agg_expected.extend(stats.expected_content)
        agg_decoded.extend(stats.decoded_content)

        write(result)
        done += 1
        torch.cuda.empty_cache()

        mass = ""
        if timing.get("decoded_content_mean") is not None:
            mass = f"  content: decoded={timing['decoded_content_mean']:.0f}"
            if timing.get("expected_content_mean") is not None:
                mass += (f" expected={timing['expected_content_mean']:.0f}"
                         f" gap={timing['content_gap']:+.0f}")
        print(f"    valid={valid} passed={passed} {elapsed:.1f}s  "
              f"fwd={timing['forward_mean_ms']:.0f}ms/step "
              f"infer={timing['infer_mean_ms']:.0f}ms/step "
              f"overhead={timing['constraint_overhead_pct']:.0f}%{mass}")

    # ── mass diagnostic, aggregated ───────────────────────────────────────
    # The go/no-go for mass-based decoding. At every point where the decoder
    # could either close the structure or keep writing, this compares the total
    # posterior mass of all content continuations against the mass of stopping.
    # Well above 50% means max-marginal and sampling can recover content that
    # Viterbi collapses; at or below 50% the posterior itself prefers to stop,
    # no decoder sampling from it will help, and the fix has to change the
    # objective (a length penalty) rather than the search.
    if agg_decoded:
        d = sum(agg_decoded) / len(agg_decoded)
        print(f"\ncontent diagnostic ({args.decoder} arm, {len(agg_decoded):,} steps)")
        print(f"  decoded  non-EOS tokens/step   {d:6.1f}")
        if agg_expected:
            e = sum(agg_expected) / len(agg_expected)
            print(f"  expected non-EOS tokens/step   {e:6.1f}   (constrained posterior)")
            print(f"  gap                            {e - d:+6.1f}")
            verdict = ("the posterior carries content this decoder discards -- "
                       "a mass-based decoder can recover it"
                       if e - d > 1.0 else
                       "the posterior itself prefers to stop; changing the decoder "
                       "cannot fix this, the objective needs a length penalty")
            print(f"  verdict: {verdict}")
        else:
            print("  expected: n/a on the viterbi arm (max-product scores are not "
                  "masses). Compare this against the marginal arm's `expected`.")

    # ── coverage ──────────────────────────────────────────────────────────
    n = len(instances)
    print(f"\n{spec.name}: decoded {done}/{n} ({100 * done / max(n, 1):.1f}% coverage)")
    for reason, count in skips.most_common():
        print(f"  skipped {count:4d}  {reason}")
    if skips:
        print("  report coverage next to validity: a method that skips hard "
              "instances is not comparable to one that answers them.")


if __name__ == "__main__":
    main()

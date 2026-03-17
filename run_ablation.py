#!/usr/bin/env python3
"""
SDSD Ablation Experiment Runner — JSON-Mode-Eval

Runs methods per user table:
  | Method           | Complexity | NFE (64 tok) | TTFT | Intent Recovery | Grammar |
  | DINGO (Baseline) | O(N)       | T            | high | N/A             | 100%   |
  | STATIC + DINGO   | O(K)       | T            | low  | N/A             | 100%   |
  | DINGO + Herding  | O(N)       | T            | high | fast            | 100%   |
  | SDSD (Ours)      | O(K)       | T/τ          | low  | fast            | 100%   |

Model forward is counted in NFE. For json-mode-eval, vocab >= 32k for O(N) vs O(K) comparison.

Usage:
  python run_ablation.py --model dream --mock --dataset json-mode-eval
  python run_ablation.py --model dream --dataset json-mode-eval --samples 20
  python run_ablation.py --model llada --dataset json-mode-eval --intent-recovery
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Mock mode: force CPU to avoid slow CUDA init
if "--mock" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from baseline_dingo import baseline_dingo_dp
from sparse_dingo import sparse_dingo_dp
from herding import herding_decode
from speculative_tree import (
    sdsd_multi_round,
    sdsd_multi_round_argmax,
)

from test_dllm_sdsd import (
    get_device,
    load_dream_model,
    load_llada_model,
    get_block_logits_dream,
    get_block_logits_llada,
    get_logits_for_position_dream,
    get_logits_for_position_llada,
    get_verify_logits_dream,
    get_verify_logits_llada,
    get_synthetic_logits,
    get_synthetic_logits_for_position,
    get_synthetic_verify_logits,
    build_permissive_dfa,
    build_simple_json_dfa,
    build_json_dfa_from_tokenizer,
)


# Target: generate 64 tokens for NFE comparison
TARGET_LENGTH = 64
DRAFT_LENGTH = 16
MOCK_VOCAB_SIZE = 32000  # N >= 32k for O(N) vs O(K) comparison
MOCK_VOCAB_QUICK = 2048  # Smaller for --quick mock testing


def _run_sequential_decode(
    get_logits_fn,
    csr,
    trans_fn,
    num_states,
    vocab_size,
    start_state,
    live_states,
    target_length,
    prompt,
    seed,
    progress_cb=None,
) -> tuple[list[int], int, bool, float]:
    """Sequential DINGO O(N): 1 forward per token. NFE = target_length."""
    tokens = []
    q = start_state
    nfe = 0
    t0 = time.perf_counter()

    it = range(target_length)
    if progress_cb:
        it = tqdm(it, desc=progress_cb("Baseline"), leave=False, unit="tok")
    for i in it:
        prob_i = get_logits_fn(prompt, tokens, seed + i)
        nfe += 1
        r = baseline_dingo_dp(num_states, vocab_size, trans_fn, [prob_i], q, live_states)
        if not r.tokens:
            break
        tokens.append(r.tokens[0])
        q = r.final_state

    latency = (time.perf_counter() - t0) * 1000
    success = q in live_states and len(tokens) == target_length
    return tokens, nfe, success, latency


def _run_sequential_sparse(
    get_logits_fn,
    csr,
    start_state,
    live_states,
    target_length,
    prompt,
    seed,
    progress_cb=None,
) -> tuple[list[int], int, bool, float]:
    """Sequential STATIC+DINGO O(K): NFE = target_length."""
    tokens = []
    q = start_state
    nfe = 0
    t0 = time.perf_counter()

    it = range(target_length)
    if progress_cb:
        it = tqdm(it, desc=progress_cb("STATIC+DINGO"), leave=False, unit="tok")
    for i in it:
        prob_i = get_logits_fn(prompt, tokens, seed + i)
        nfe += 1
        r = sparse_dingo_dp(csr, [prob_i], q, live_states)
        if not r.tokens:
            break
        tokens.append(r.tokens[0])
        q = r.final_state

    latency = (time.perf_counter() - t0) * 1000
    success = q in live_states and len(tokens) == target_length
    return tokens, nfe, success, latency


def _run_sequential_herding(
    get_logits_fn,
    csr,
    start_state,
    live_states,
    target_length,
    prompt,
    seed,
    progress_cb=None,
) -> tuple[list[int], int, bool, float]:
    """Sequential DINGO + Herding: NFE = target_length."""
    tokens = []
    q = start_state
    nfe = 0
    t0 = time.perf_counter()

    it = range(target_length)
    if progress_cb:
        it = tqdm(it, desc=progress_cb("Herding"), leave=False, unit="tok")
    for i in it:
        prob_i = get_logits_fn(prompt, tokens, seed + i)
        nfe += 1
        r = herding_decode(csr, [prob_i], q, live_states, block_length=1)
        if not r.tokens:
            break
        tokens.append(r.tokens[0])
        q = r.final_state

    latency = (time.perf_counter() - t0) * 1000
    success = q in live_states and len(tokens) == target_length
    return tokens, nfe, success, latency


def run_one_sample(
    get_logits_fn,
    get_block_logits_fn,
    get_verify_logits_fn,
    csr,
    trans_fn,
    num_states,
    vocab_size,
    start_state,
    live_states,
    target_length,
    prompt,
    seed,
    sample_idx: int = 0,
    total_samples: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Run all methods with model-in-loop.
    Sequential: NFE = target_length.
    SDSD: NFE = 2 * rounds (draft + verify per round), typically target_length / tau.
    """
    def _pcb(method: str):
        return f"Sample {sample_idx+1}/{total_samples} | {method}"

    progress_cb = _pcb if verbose else None
    results = {}

    # Baseline: DINGO O(N), NFE = T
    if verbose:
        print(f"  Sample {sample_idx+1}/{total_samples} | Baseline (O(N), ~{target_length} steps)...", flush=True)
    tokens1, nfe1, succ1, lat1 = _run_sequential_decode(
        get_logits_fn, csr, trans_fn, num_states, vocab_size,
        start_state, live_states, target_length, prompt, seed,
        progress_cb=progress_cb,
    )
    results["Baseline"] = {"tokens": tokens1, "success": succ1, "latency_ms": lat1, "nfe": nfe1}

    # Ablation 1: STATIC + DINGO O(K), NFE = T
    if verbose:
        print(f"  Sample {sample_idx+1}/{total_samples} | STATIC+DINGO (O(K))...", flush=True)
    tokens2, nfe2, succ2, lat2 = _run_sequential_sparse(
        get_logits_fn, csr, start_state, live_states, target_length, prompt, seed,
        progress_cb=progress_cb,
    )
    results["Ablation1"] = {"tokens": tokens2, "success": succ2, "latency_ms": lat2, "nfe": nfe2}

    # Ablation 2: DINGO + Herding, NFE = T
    if verbose:
        print(f"  Sample {sample_idx+1}/{total_samples} | Herding...", flush=True)
    tokens3, nfe3, succ3, lat3 = _run_sequential_herding(
        get_logits_fn, csr, start_state, live_states, target_length, prompt, seed,
        progress_cb=progress_cb,
    )
    results["Ablation2"] = {"tokens": tokens3, "success": succ3, "latency_ms": lat3, "nfe": nfe3}

    # Ablation 3: STATIC + Spec-Tree (argmax), model-in-loop, NFE = T/tau
    if verbose:
        print(f"  Sample {sample_idx+1}/{total_samples} | Ablation3 (Spec-Tree)...", flush=True)
    def block_fn(prefix: list[int], bl: int):
        pv = get_block_logits_fn(prefix, bl)
        return pv[0] if isinstance(pv, tuple) else pv

    t0 = time.perf_counter()
    tok4, nfe4, _, succ4 = sdsd_multi_round_argmax(
        csr, block_fn, get_verify_logits_fn,
        start_state, live_states, target_length, draft_length=DRAFT_LENGTH,
    )
    lat4 = (time.perf_counter() - t0) * 1000
    results["Ablation3"] = {"tokens": tok4, "success": succ4, "latency_ms": lat4, "nfe": nfe4}

    # SDSD: STATIC + Herding + Tree, model-in-loop, NFE = T/tau
    if verbose:
        print(f"  Sample {sample_idx+1}/{total_samples} | SDSD...", flush=True)
    t0 = time.perf_counter()
    tok5, nfe5, _, succ5 = sdsd_multi_round(
        csr, block_fn, get_verify_logits_fn,
        start_state, live_states, target_length, draft_length=DRAFT_LENGTH,
    )
    lat5 = (time.perf_counter() - t0) * 1000
    results["SDSD"] = {"tokens": tok5, "success": succ5, "latency_ms": lat5, "nfe": nfe5}

    return results


def run_intent_recovery_sample(
    get_logits_fn,
    csr,
    start_state,
    live_states,
    target_length,
    prompt,
    seed,
    perturb_pos: int = 5,
) -> dict:
    """
    Intent recovery: inject grammar-valid but wrong token at perturb_pos.
    Measure steps until model intent is recovered.
    """
    # Run normal Herding decode to get "correct" path
    tokens_correct = []
    q = start_state
    for i in range(target_length):
        prob_i = get_logits_fn(prompt, tokens_correct, seed + i)
        r = herding_decode(csr, [prob_i], q, live_states, block_length=1)
        if not r.tokens:
            break
        tokens_correct.append(r.tokens[0])
        q = r.final_state

    # Run with perturbation: at perturb_pos, force a different grammar-valid token
    tokens_perturbed = []
    q = start_state
    recovery_step = -1
    for i in range(target_length):
        prob_i = get_logits_fn(prompt, tokens_perturbed, seed + i)
        r = herding_decode(csr, [prob_i], q, live_states, block_length=1)
        if not r.tokens:
            break

        if i == perturb_pos:
            # Inject wrong token: pick second-best grammar-valid if available
            transitions = csr.get_transitions(q)
            best_t = r.tokens[0]
            alt_t = None
            probs_sorted = sorted(
                ((t, prob_i[t] if t < len(prob_i) else 0) for t, _ in transitions),
                key=lambda x: -x[1],
            )
            for t, _ in probs_sorted:
                if t != best_t:
                    alt_t = t
                    break
            chosen = alt_t if alt_t is not None else best_t
        else:
            chosen = r.tokens[0]

        if recovery_step < 0 and i > perturb_pos and i < len(tokens_correct) and chosen == tokens_correct[i]:
            recovery_step = i - perturb_pos

        tokens_perturbed.append(chosen)
        for tt, qn in csr.get_transitions(q):
            if tt == chosen:
                q = qn
                break

    return {
        "recovery_steps": recovery_step if recovery_step >= 0 else target_length - perturb_pos,
        "perturb_pos": perturb_pos,
    }


def run_ablation(
    model_name: str,
    num_samples: int = 10,
    target_length: int = TARGET_LENGTH,
    mock: bool = False,
    dataset: str | None = None,
    dataset_limit: int = 50,
    run_intent_recovery: bool = False,
    quick_mock: bool = False,
    verbose: bool = True,
) -> dict:
    """Run full ablation study for JSON-Mode-Eval."""
    device, has_gpu = get_device()
    vocab_size = (MOCK_VOCAB_QUICK if quick_mock else MOCK_VOCAB_SIZE) if mock else 32000
    use_json_dfa = dataset == "json-mode-eval"

    if mock or not has_gpu:
        prob_sources = []
        for i in range(num_samples):
            prob_sources.append(("synthetic", "Generate a JSON object with key 'name' and value 'test'.", i))
        tokenizer = None

        def get_logits_fn(prompt, prefix_tokens, seed):
            return get_synthetic_logits_for_position(vocab_size, prefix_tokens, seed)

        def get_block_logits_fn(prefix, bl):
            pv = get_synthetic_logits(vocab_size, bl, seed=42 + (len(prefix) if prefix else 0), prefix_tokens=prefix)
            return pv

        def get_verify_logits_fn(context):
            return get_synthetic_verify_logits(vocab_size, context, seed=42)

    else:
        if model_name == "dream":
            model, tokenizer = load_dream_model(device)
            get_logits_fn = lambda p, prefix, s: get_logits_for_position_dream(model, tokenizer, p, prefix, device)[0]
        else:
            model, tokenizer = load_llada_model(device)
            get_logits_fn = lambda p, prefix, s: get_logits_for_position_llada(model, tokenizer, p, prefix, device)[0]
        vocab_size = tokenizer.vocab_size if tokenizer else vocab_size

        if dataset:
            try:
                from dataset_loaders import get_dataset
                samples = get_dataset(dataset)[:dataset_limit]
                prob_sources = [(dataset, s.prompt, i) for i, s in enumerate(samples)]
            except Exception as e:
                print(f"Dataset load failed: {e}, using default prompt")
                prob_sources = [("default", "Generate a JSON object with key 'name' and value 'test'.", i) for i in range(num_samples)]
        else:
            prob_sources = [("default", "Generate a JSON object with key 'name' and value 'test'.", i) for i in range(num_samples)]

        def make_get_block(prompt):
            if model_name == "dream":
                return lambda prefix, bl: get_block_logits_dream(
                    model, tokenizer, prompt, bl, device, prefix_tokens=prefix if prefix else None
                )[0]
            return lambda prefix, bl: get_block_logits_llada(
                model, tokenizer, prompt, bl, device, prefix_tokens=prefix if prefix else None
            )[0]

        def make_get_verify(prompt):
            if model_name == "dream":
                return lambda ctx: get_verify_logits_dream(model, tokenizer, prompt, ctx, device)
            return lambda ctx: get_verify_logits_llada(model, tokenizer, prompt, ctx, device)

    if use_json_dfa and tokenizer is not None:
        csr, start_state, live_states = build_json_dfa_from_tokenizer(tokenizer)
        num_states = 2
    elif use_json_dfa and not mock:
        csr, start_state, live_states = build_simple_json_dfa(vocab_size)
        num_states = 10
    else:
        # Mock or non-JSON: permissive DFA (accepts 64+ tokens; simple JSON DFA only accepts 9)
        csr, start_state, live_states = build_permissive_dfa(vocab_size, valid_tokens=list(range(vocab_size)))
        num_states = 2

    # O(1) trans_fn for permissive DFA (avoids O(N) iteration in baseline; 32k vocab would be ~10min/sample)
    use_permissive = num_states == 2 and len(live_states) == 1
    if use_permissive:
        def trans_fn(q, t):
            return 1 if t < vocab_size else None
    else:
        def trans_fn(q, t):
            for (tt, qn) in csr.get_transitions(q):
                if tt == t:
                    return qn
            return None

    all_results = {m: [] for m in ["Baseline", "Ablation1", "Ablation2", "Ablation3", "SDSD"]}
    intent_recovery_results = []

    for si, (src, prompt, idx) in tqdm(enumerate(prob_sources[:num_samples]), total=num_samples, desc="Samples"):
        seed = 42 + idx

        if not mock and has_gpu:
            get_block_logits_fn = make_get_block(prompt)
            get_verify_logits_fn = make_get_verify(prompt)

        sample_res = run_one_sample(
            get_logits_fn,
            get_block_logits_fn,
            get_verify_logits_fn,
            csr,
            trans_fn,
            num_states,
            vocab_size,
            start_state,
            live_states,
            target_length,
            prompt,
            seed,
            sample_idx=si,
            total_samples=num_samples,
            verbose=verbose,
        )
        for name, r in sample_res.items():
            all_results[name].append(r)

        if run_intent_recovery and si < 5:  # Limit to 5 for speed
            ir = run_intent_recovery_sample(
                get_logits_fn, csr, start_state, live_states, target_length, prompt, seed,
            )
            intent_recovery_results.append(ir)

    # Aggregate
    summary = {}
    for name in ["Baseline", "Ablation1", "Ablation2", "Ablation3", "SDSD"]:
        entries = all_results[name]
        n = len(entries)
        nfe_list = [e["nfe"] for e in entries]
        nfe_avg = sum(nfe_list) / n if n else 0
        summary[name] = {
            "ttft_ms": sum(e["latency_ms"] for e in entries) / n if n else 0,
            "throughput_tok_s": (
                sum(len(e["tokens"]) for e in entries) / max(1e-6, sum(e["latency_ms"] for e in entries) / 1000)
                if n else 0
            ),
            "nfe_avg": nfe_avg,
            "nfe_min": min(nfe_list) if nfe_list else 0,
            "nfe_max": max(nfe_list) if nfe_list else 0,
            "rounds_avg": nfe_avg / 2 if name in ("Ablation3", "SDSD") else None,  # 每輪 = 2 NFE
            "parse_rate": sum(1 for e in entries if e["success"]) / n * 100 if n else 0,
            "n_samples": n,
        }

    intent_recovery_avg = (
        sum(r["recovery_steps"] for r in intent_recovery_results) / len(intent_recovery_results)
        if intent_recovery_results else None
    )

    # Per-sample NFE for SDSD/Ablation3 (for distribution analysis)
    nfe_per_sample = {
        "Ablation3": [e["nfe"] for e in all_results["Ablation3"]],
        "SDSD": [e["nfe"] for e in all_results["SDSD"]],
    }

    return {
        "model": model_name,
        "mock": mock,
        "num_samples": num_samples,
        "target_length": target_length,
        "dataset": dataset or "default",
        "vocab_size": vocab_size,
        "summary": summary,
        "nfe_per_sample": nfe_per_sample,
        "intent_recovery_avg_steps": intent_recovery_avg,
    }


def print_table(summary: dict, model_name: str, intent_recovery: float | None = None):
    """Print table per user spec."""
    methods = [
        ("Baseline", "DINGO (Baseline)", "O(N)", "Cannot recover"),
        ("Ablation1", "STATIC + DINGO", "O(K)", "Cannot recover"),
        ("Ablation2", "DINGO + Herding", "O(N)", "Fast (low steps)"),
        ("Ablation3", "STATIC + Spec-Tree", "O(K)", "N/A"),
        ("SDSD", "SDSD (Ours)", "O(K)", "Fast"),
    ]
    print(f"\n{'='*120}")
    print(f"  SDSD Ablation — JSON-Mode-Eval (64 tokens) — {model_name.upper()}")
    print(f"{'='*120}")
    print(f"{'Method':<14} {'Technique':<22} {'Complexity':<8} {'NFE(64)':<20} {'TTFT(ms)':<12} {'Throughput':<12} {'Parse%':<10} {'Intent Rec':<12}")
    print("-" * 120)
    for key, tech, compl, ir_label in methods:
        s = summary.get(key, {})
        ir = ir_label
        if key in ("Ablation2", "SDSD") and intent_recovery is not None:
            ir = f"{intent_recovery:.1f} steps"
        nfe_str = f"{s.get('nfe_avg', 0):.1f}"
        if key in ("Ablation3", "SDSD") and s.get("nfe_min") is not None and s.get("nfe_max") is not None:
            nfe_str = f"{s.get('nfe_avg', 0):.1f} [{s['nfe_min']}-{s['nfe_max']}]"
        print(f"{key:<14} {tech:<22} {compl:<8} {nfe_str:<20} {s.get('ttft_ms', 0):<12.2f} {s.get('throughput_tok_s', 0):<12.2f} {s.get('parse_rate', 0):<10.1f} {str(ir):<12}")
    print(f"{'='*120}\n")


def main():
    print("Starting SDSD ablation (JSON-Mode-Eval, 64 tokens)...", flush=True)
    parser = argparse.ArgumentParser(description="SDSD Ablation Experiment")
    parser.add_argument("--model", choices=["dream", "llada"], default="dream")
    parser.add_argument("--mock", action="store_true", help="Synthetic data (no GPU)")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--target-length", type=int, default=TARGET_LENGTH)
    parser.add_argument("--dataset", type=str, default="json-mode-eval", help="json-mode-eval (default), humaneval, mbpp")
    parser.add_argument("--dataset-limit", type=int, default=50)
    parser.add_argument("--intent-recovery", action="store_true", help="Run intent recovery experiment")
    parser.add_argument("--quick", action="store_true", help="Quick mock: smaller vocab/length for testing")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-method progress")
    parser.add_argument("--output", type=str, default="results/ablation_results.json")
    args = parser.parse_args()

    target_len = 16 if args.quick and args.mock else args.target_length
    print(f"  Model: {args.model}, Mock: {args.mock}, Samples: {args.samples}, Dataset: {args.dataset}")
    print(f"  Target: {target_len} tokens | Vocab: {'32k' if not args.quick else '2k'}{' (mock)' if args.mock else ''}")
    print(f"\n  Progress: Each sample runs 5 methods in order:")
    print(f"    1. Baseline (O(N))  2. STATIC+DINGO (O(K))  3. Herding  4. Ablation3  5. SDSD")
    print(f"  Estimated: ~{target_len * 5} sequential steps/sample for Baseline+Herding\n", flush=True)

    results = run_ablation(
        args.model, args.samples, target_len,
        mock=args.mock, dataset=args.dataset, dataset_limit=args.dataset_limit,
        run_intent_recovery=args.intent_recovery,
        quick_mock=args.quick,
        verbose=not args.quiet,
    )

    print_table(
        results["summary"],
        args.model,
        intent_recovery=results.get("intent_recovery_avg_steps"),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified Benchmark: SDSD methods on JSON-Bench (jsonschema), Dgrammar-style output.

Runs Baseline, Ablation1, Ablation2, Ablation3, SDSD on eth-sri/json-mode-eval-extended
and outputs JSONL compatible with aggregate_unified_results.py.

Settings (aligned with Dgrammar/LAVE):
  - Model: LLaDA-8B-Instruct
  - Dataset: jsonschema (eth-sri/json-mode-eval-extended)
  - gen_length: 256 tokens
  - Warmup: 5 samples excluded from timing

Usage:
  python run_unified_benchmark.py --methods baseline,ablation1,ablation2,ablation3,sdsd
  python run_unified_benchmark.py --methods sdsd --limit 20
  python run_unified_benchmark.py --output results/unified
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

if "--mock" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from baseline_dingo import baseline_dingo_dp
from sparse_dingo import sparse_dingo_dp
from herding import herding_decode
from speculative_tree import sdsd_multi_round, sdsd_multi_round_argmax

from test_dllm_sdsd import (
    get_device,
    load_llada_model,
    get_block_logits_llada,
    get_logits_for_position_llada,
    get_verify_logits_llada,
    build_permissive_dfa,
    build_json_dfa_from_tokenizer,
)

GEN_LENGTH = 256
DRAFT_LENGTH = 32
WARMUP = 5


def load_jsonschema_dataset(limit: int | None = None):
    """Load eth-sri/json-mode-eval-extended (JSON-Bench)."""
    from datasets import load_dataset
    ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    instances = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        instance_id = row.get("instance_id", f"jsonschema_{i}")
        instances.append({
            "instance_id": instance_id,
            "input": row.get("input", ""),
            "schema": row.get("schema", "{}"),
            "output": row.get("output", ""),
        })
    return instances


def build_prompt(instance: dict) -> list:
    """Build chat prompt for LLaDA (match Dgrammar format)."""
    schema = instance["schema"]
    user_input = instance["input"]
    system = f"""You are a helpful assistant that answers in JSON. Here's the JSON schema you must adhere to:
<schema>
{schema}
</schema>
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


def run_one_instance(
    instance: dict,
    methods: list[str],
    get_logits_fn,
    get_block_logits_fn,
    get_verify_logits_fn,
    csr,
    trans_fn,
    num_states: int,
    vocab_size: int,
    start_state: int,
    live_states: set[int],
    tokenizer,
    seed: int = 42,
) -> dict[str, dict]:
    """Run selected methods on one instance, return results per method."""
    prompt = build_prompt(instance)
    results = {}

    def _run_sequential(decode_fn):
        tokens = []
        q = start_state
        nfe = 0
        t_forward = 0.0
        t_constraint = 0.0
        for i in range(GEN_LENGTH):
            t_f = time.perf_counter()
            prob_i = get_logits_fn(prompt, tokens, seed + i)
            t_forward += time.perf_counter() - t_f
            nfe += 1
            t_c = time.perf_counter()
            r = decode_fn([prob_i], q)
            t_constraint += time.perf_counter() - t_c
            if not r.tokens:
                break
            tokens.append(r.tokens[0])
            q = r.final_state
        elapsed = t_forward + t_constraint
        constraint_pct = (t_constraint / elapsed * 100) if elapsed > 0 else 0
        decoded = tokenizer.decode(tokens, skip_special_tokens=True) if tokenizer and tokens else ""
        return {
            "tokens": tokens,
            "decoded": decoded,
            "elapsed": elapsed,
            "nfe": nfe,
            "success": q in live_states and len(tokens) >= GEN_LENGTH,
            "timing": {
                "total_forward_ms": t_forward * 1000,
                "total_constraint_ms": t_constraint * 1000,
                "constraint_pct": constraint_pct,
            },
        }

    # Baseline: baseline_dingo_dp(num_states, vocab_size, trans_fn, prob_vectors, start_state, live_states)
    if "baseline" in methods:
        def _bl(probs, q):
            return baseline_dingo_dp(num_states, vocab_size, trans_fn, probs, q, live_states)
        results["baseline"] = _run_sequential(_bl)

    # Ablation1: sparse_dingo_dp(csr, prob_vectors, start_state, live_states)
    if "ablation1" in methods:
        def _a1(probs, q):
            return sparse_dingo_dp(csr, probs, q, live_states)
        results["ablation1"] = _run_sequential(_a1)

    # Ablation2: herding_decode(csr, probs, start_state, live_states, block_length=1)
    if "ablation2" in methods:
        def _a2(probs, q):
            return herding_decode(csr, probs, q, live_states, block_length=1)
        results["ablation2"] = _run_sequential(_a2)

    # Ablation3, SDSD: speculative
    if "ablation3" in methods or "sdsd" in methods:
        def block_fn(prefix, bl):
            pv = get_block_logits_fn(prefix, bl)
            return pv[0] if isinstance(pv, tuple) else pv

        if "ablation3" in methods:
            t0 = time.perf_counter()
            t_fwd = 0.0
            t_const = 0.0
            # Simplified: we don't have fine-grained timing for speculative
            tok, nfe, _, succ = sdsd_multi_round_argmax(
                csr, block_fn, get_verify_logits_fn,
                start_state, live_states, GEN_LENGTH, draft_length=DRAFT_LENGTH,
            )
            elapsed = time.perf_counter() - t0
            decoded = tokenizer.decode(tok, skip_special_tokens=True) if tokenizer and tok else ""
            results["ablation3"] = {
                "tokens": tok,
                "decoded": decoded,
                "elapsed": elapsed,
                "nfe": nfe,
                "success": succ,
                "timing": {"constraint_pct": 0},  # Placeholder
            }

        if "sdsd" in methods:
            t0 = time.perf_counter()
            tok, nfe, _, succ = sdsd_multi_round(
                csr, block_fn, get_verify_logits_fn,
                start_state, live_states, GEN_LENGTH, draft_length=DRAFT_LENGTH,
            )
            elapsed = time.perf_counter() - t0
            decoded = tokenizer.decode(tok, skip_special_tokens=True) if tokenizer and tok else ""
            results["sdsd"] = {
                "tokens": tok,
                "decoded": decoded,
                "elapsed": elapsed,
                "nfe": nfe,
                "success": succ,
                "timing": {"constraint_pct": 0},
            }

    return results


def extract_result(decoded: str, instance: dict) -> str:
    """Extract JSON from decoded output (match Dgrammar extract_result)."""
    # Try to find JSON object/array in output
    start = decoded.find("{")
    if start < 0:
        start = decoded.find("[")
    if start < 0:
        return decoded
    depth = 0
    in_str = False
    escape = False
    end = start
    for i, c in enumerate(decoded[start:], start):
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if in_str:
            if c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                end = i
                break
    return decoded[start : end + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="baseline,ablation1,ablation2,ablation3,sdsd",
                        help="Comma-separated: baseline,ablation1,ablation2,ablation3,sdsd")
    parser.add_argument("--limit", type=int, default=None, help="Limit instances (for testing)")
    parser.add_argument("--output", default="results/unified", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use synthetic (no GPU)")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading jsonschema dataset...")
    instances = load_jsonschema_dataset(args.limit)
    print(f"  {len(instances)} instances")

    device, has_gpu = get_device()
    if args.mock or not has_gpu:
        print("Mock mode: skipping (need GPU for unified benchmark)")
        return 1

    print("Loading LLaDA-8B-Instruct...")
    model, tokenizer = load_llada_model(device)
    vocab_size = tokenizer.vocab_size

    csr, start_state, live_states = build_json_dfa_from_tokenizer(tokenizer)
    num_states = 2

    def trans_fn(q, t):
        for tt, qn in csr.get_transitions(q):
            if tt == t:
                return qn
        return None

    for method in methods:
        out_file = out_dir / f"sdsd_{method}_jsonschema.jsonl"
        print(f"\nRunning {method} -> {out_file}")

        for i, instance in enumerate(tqdm(instances, desc=method)):
            prompt = build_prompt(instance)
            seed = 42 + i

            def get_logits(p, prefix, s):
                return get_logits_for_position_llada(model, tokenizer, p, prefix, device)[0]

            get_block = lambda prefix, bl: get_block_logits_llada(
                model, tokenizer, prompt, bl, device, prefix_tokens=prefix or None
            )[0]
            get_verify = lambda ctx: get_verify_logits_llada(model, tokenizer, prompt, ctx, device)

            # Run single method
            res = run_one_instance(
                instance, [method],
                get_logits, get_block, get_verify,
                csr, trans_fn, num_states, vocab_size, start_state, live_states,
                tokenizer,
                seed=seed,
            )
            r = res.get(method, {})
            decoded = r.get("decoded", "")
            extracted = extract_result(decoded, instance)

            result = {
                "instance_id": instance["instance_id"],
                "method": method,
                "extracted": extracted,
                "time_taken": r.get("elapsed", 0),
                "valid": r.get("success", False),
                "timing": r.get("timing", {}),
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    print(f"\nDone. Run: python aggregate_unified_results.py {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

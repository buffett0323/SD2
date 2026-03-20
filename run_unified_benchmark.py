#!/usr/bin/env python3
"""
Unified Benchmark: SDSD constraint methods on LLaDA-8B (diffusion model).

Structure (run_unified_benchmark.sh):
  - Dgrammar/LAVE/IG-CD: run via vendor/dgrammar (diffusion T=128, their constraint)
  - SDSD: run via this script (diffusion T=128, our DINGO/Herding at frontier)

All on same model (LLaDA-8B), same dataset (JSON-Bench 272). All use diffusion (T=128).
SDSD replaces Dgrammar's argmax at frontier with our DINGO/Herding constrained decoding.

Usage:
  python run_unified_benchmark.py --methods sdsd,ablation1,ablation2,ablation3
  python run_unified_benchmark.py --methods sdsd --limit 20
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
# For schema_guided: TokenChecker from vendor/dgrammar (requires llguidance)
_vendor_dgrammar = Path(__file__).resolve().parent / "vendor" / "dgrammar"
if _vendor_dgrammar.exists():
    sys.path.insert(0, str(_vendor_dgrammar))

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
    build_json_dfa_from_tokenizer,
)

GEN_LENGTH = 256
DRAFT_LENGTH = 32
WARMUP = 5

# schema_guided (AR + llguidance) requires vendor/dgrammar
_SCHEMA_GUIDED_AVAILABLE = False
try:
    from dgrammar.checker import TokenChecker
    _SCHEMA_GUIDED_AVAILABLE = True
except ImportError:
    pass


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


def prompt_to_ids(prompt: list, tokenizer, device) -> tuple["torch.Tensor", int]:
    """Convert chat prompt to token ids for diffusion. Returns (prompt_ids, prompt_len)."""
    import torch
    prompt_str = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)
    return input_ids, input_ids.shape[1]


def _run_diffusion_sdsd(
    instance: dict,
    model,
    tokenizer,
    method: str,
    device,
    gen_length: int = 256,
    steps: int = 128,
    block_length: int = 32,
) -> dict:
    """Run SDSD diffusion: our DINGO/Herding at frontier, same T/steps as Dgrammar."""
    from diffusion_sdsd import generate_diffusion_sdsd, make_frontier_picker
    from herding import HerdingMomentumState

    prompt = build_prompt(instance)
    prompt_ids, prompt_len = prompt_to_ids(prompt, tokenizer, device)
    checker = TokenChecker(instance["schema"])
    # ablation2: one Herding w vector for the whole diffusion run (cross-step momentum)
    herding_state = HerdingMomentumState() if method == "ablation2" else None
    frontier_picker = make_frontier_picker(method, herding_state=herding_state)

    bidi_csr = bidi_live = bidi_start = None
    if method == "bidi":
        bidi_csr, bidi_start, bidi_live = build_json_dfa_from_tokenizer(tokenizer)

    mask_id = getattr(model.config, "mask_token_id", None) or 126336
    eos_id = 126081
    eot_id = 126348

    t0 = time.perf_counter()
    out = None
    valid = False
    for out, resamples, valid, _v, _r, _gc in generate_diffusion_sdsd(
        model, prompt_ids, tokenizer, checker,
        prompt_len=prompt_len,
        frontier_picker=frontier_picker,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.2,
        mask_id=mask_id,
        eos_id=eos_id,
        eot_id=eot_id,
        bidi_csr=bidi_csr,
        bidi_live_states=bidi_live,
        bidi_json_start_state=bidi_start if bidi_start is not None else 0,
    ):
        pass
    elapsed = time.perf_counter() - t0

    if out is None:
        decoded = ""
    else:
        gen_start = prompt_ids.shape[1]
        gen_ids = out[0, gen_start:].tolist()
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "tokens": [],
        "decoded": decoded,
        "elapsed": elapsed,
        "nfe": steps,
        "success": valid,
        "timing": {"constraint_pct": 0},
    }


def _run_schema_guided_ar(
    instance: dict,
    get_logits_fn,
    checker: "TokenChecker",
    tokenizer,
    vocab_size: int,
    seed: int = 42,
) -> dict:
    """Schema-specific constrained decode via llguidance (correct output)."""
    import torch
    prompt = build_prompt(instance)
    tokens = []
    t_forward = 0.0
    t_constraint = 0.0
    checker.reset()
    for i in range(GEN_LENGTH):
        if checker.is_accepting():
            break
        t_f = time.perf_counter()
        logits = get_logits_fn(prompt, tokens, seed + i)
        t_forward += time.perf_counter() - t_f
        t_c = time.perf_counter()
        bias = checker.compute_mask(vocab_size=vocab_size)
        t_constraint += time.perf_counter() - t_c
        if isinstance(logits, list):
            logits_t = torch.tensor(logits, dtype=torch.float32)
        else:
            logits_t = logits if hasattr(logits, "shape") else torch.tensor(logits, dtype=torch.float32)
        if hasattr(logits_t, "cpu"):
            logits_t = logits_t.cpu()
        if bias.shape[0] > logits_t.shape[0]:
            bias = bias[: logits_t.shape[0]]
        elif bias.shape[0] < logits_t.shape[0]:
            pad = torch.ones(logits_t.shape[0] - bias.shape[0], dtype=torch.bool)
            bias = torch.cat([bias, pad])
        logits_t = logits_t.clone()
        logits_t[bias] = float("-inf")
        best = int(torch.argmax(logits_t).item())
        if logits_t[best] == float("-inf"):
            break
        c = checker.matcher.try_consume_tokens([best])
        if c != 1:
            break
        tokens.append(best)
    elapsed = t_forward + t_constraint
    constraint_pct = (t_constraint / elapsed * 100) if elapsed > 0 else 0
    decoded = tokenizer.decode(tokens, skip_special_tokens=True) if tokenizer and tokens else ""
    return {
        "tokens": tokens,
        "decoded": decoded,
        "elapsed": elapsed,
        "nfe": len(tokens),
        "success": checker.is_accepting(),
        "timing": {
            "total_forward_ms": t_forward * 1000,
            "total_constraint_ms": t_constraint * 1000,
            "constraint_pct": constraint_pct,
        },
    }


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

    # Baseline: baseline_dingo_dp — O(N) per step, SLOW with strict DFA
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


SLOW_METHODS = {"baseline"}  # O(N) per step, ~30–60 min/instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="sdsd,ablation1,ablation2,ablation3",
                        help="Comma-separated: sdsd,ablation1,ablation2,ablation3,baseline,schema_guided")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit instances (e.g. --limit 5 for quick test; 272 full)")
    parser.add_argument("--output", default="results/unified", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use synthetic (no GPU)")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow methods (baseline). Use for faster runs.")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_slow:
        methods = [m for m in methods if m not in SLOW_METHODS]
        if any(m in SLOW_METHODS for m in args.methods.split(",")):
            print(f"Skipping slow methods: {SLOW_METHODS}")

    if "schema_guided" in methods and not _SCHEMA_GUIDED_AVAILABLE:
        print("schema_guided requires llguidance. Install: pip install llguidance>=1.6")
        methods = [m for m in methods if m != "schema_guided"]

    diffusion_methods = {"sdsd", "ablation1", "ablation2", "ablation3", "baseline", "argmax", "bidi"}
    if diffusion_methods & set(methods) and not _SCHEMA_GUIDED_AVAILABLE:
        print("SDSD diffusion requires dgrammar/llguidance. Install: pip install llguidance>=1.6")
        methods = [m for m in methods if m not in diffusion_methods]

    if not methods:
        print("No methods to run. Add --methods or install llguidance for schema_guided.")
        return 1

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
    num_states = csr.num_states

    # Standard: use actual CSR transitions (no override). Permissive DFA has ~200 valid tokens.
    # baseline is O(N) per step → ~30–60 min/instance with 126k vocab; use --skip-slow to skip.
    def trans_fn(q, t):
        for tt, qn in csr.get_transitions(q):
            if tt == t:
                return qn
        return None

    n_inst = len(instances)
    has_slow = any(m in SLOW_METHODS for m in methods)
    est_per_instance = 45 if has_slow else (15 if any(m in methods for m in ["ablation1", "ablation2"]) else 5)
    print(f"\nMethods: {methods}")
    print(f"Estimated: ~{est_per_instance * n_inst / 60:.0f} min total")
    if SLOW_METHODS & set(methods):
        print(f"  SLOW (use --skip-slow to skip): {list(SLOW_METHODS & set(methods))}")

    for method in methods:
        out_file = out_dir / f"sdsd_{method}_jsonschema.jsonl"
        print(f"\nRunning {method} -> {out_file}")

        pbar = tqdm(instances, desc=method, unit="inst", total=len(instances),
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for i, instance in enumerate(pbar):
            prompt = build_prompt(instance)
            seed = 42 + i

            def get_logits(p, prefix, s):
                return get_logits_for_position_llada(model, tokenizer, p, prefix, device)[0]

            get_block = lambda prefix, bl: get_block_logits_llada(
                model, tokenizer, prompt, bl, device, prefix_tokens=prefix or None
            )[0]
            get_verify = lambda ctx: get_verify_logits_llada(model, tokenizer, prompt, ctx, device)

            if method == "schema_guided":
                try:
                    checker = TokenChecker(instance["schema"])
                    r = _run_schema_guided_ar(
                        instance, get_logits, checker, tokenizer, vocab_size, seed=seed
                    )
                except Exception as e:
                    pbar.write(f"  {instance['instance_id']}: schema_guided failed: {e}")
                    r = {"decoded": "", "elapsed": 0, "success": False, "timing": {}}
            elif method in ("sdsd", "ablation1", "ablation2", "ablation3", "baseline", "argmax", "bidi"):
                try:
                    r = _run_diffusion_sdsd(
                        instance, model, tokenizer, method, device,
                        gen_length=GEN_LENGTH, steps=128, block_length=32,
                    )
                except Exception as e:
                    pbar.write(f"  {instance['instance_id']}: {method} failed: {e}")
                    r = {"decoded": "", "elapsed": 0, "success": False, "timing": {}}
            else:
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

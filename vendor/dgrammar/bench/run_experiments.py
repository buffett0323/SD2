"""Unified experiment runner for dgrammar paper experiments.

Experiment A — top-k probability mass coverage
  At each violator position: what fraction of grammar-valid prob mass is in top-k?
  Validates the "top-50 covers >99% mass" claim.

Experiment B — batch size (AIMD cap) ablation
  Run with max_batch_size in {1, 2, 4, 8} and record validity + speed.
  Validates that AIMD adaptive commit with cap=8 is sufficient.

Experiment C — async overlap (already in run_dgrammar_timed.py, re-used here)
  Records: forward_ms, dp_ms (mask_compute_ms), wait_ms, overlap_count.
  Validates that DP time < GPU forward time (overlap achieved).

Experiment D — overall validity comparison
  dgrammar vs unconstrained baseline on jsb_medium.

Usage:
    python run_experiments.py <exp> [seed] [limit] [dataset] [steps] [offset] [extra...]

  exp: A | B | C | D
  extra for B: max_batch_size (int, default 8)

Output: results/exp_{exp}_{tag}_{dataset}_s{seed}_t{steps}.jsonl
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
import jsb_dataset  # noqa: F401
from dgrammar.checker import TokenChecker
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens, extend_prefix


# ── Shared helpers ────────────────────────────────────────────────────────────

mask_id = 126336
eos_id  = 126081
eot_id  = 126348


def _is_valid(out, gen_start):
    if out is None:
        return False
    gen_ids = out[0, gen_start:].tolist()
    if eos_id in gen_ids or eot_id in gen_ids:
        eos_pos = next((j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None)
        return eos_pos is not None and mask_id not in gen_ids[:eos_pos]
    return False


def _load_resources(dataset_name):
    dataset   = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    tokenizer = eval_model.tokenizer("cuda")
    model     = eval_model.model("cuda")
    return dataset, eval_model, tokenizer, model


# ── Experiment A: top-k coverage ─────────────────────────────────────────────

@torch.no_grad()
def _generate_with_coverage_probe(
    model, prompt, checker, prompt_len,
    steps=128, gen_length=256, block_length=32,
    temperature=0.2, max_batch_size=8, max_resamples=100,
):
    """Run dgrammar generation; at each violator record top-k coverage stats."""
    coverage_records = []  # list of dicts per violator position

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks      = gen_length // block_length
    steps_per_block = steps // num_blocks
    gen_start       = prompt.shape[1]
    consume_idx     = gen_start
    current_batch   = 1
    resamples       = []

    if prompt_len < gen_start:
        checker.consume_tokens(x[0, prompt_len:gen_start].tolist())

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end   = gen_start + (num_block + 1) * block_length
        block_mask_index   = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            logits             = model(x).logits
            logits_with_noise  = add_gumbel_noise(logits, temperature=temperature)
            n_scheduled        = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed = 0
            while tokens_placed < n_scheduled:
                if complete:
                    break

                mask_index = x == mask_id
                x0         = torch.argmax(logits_with_noise, dim=-1)
                p          = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p       = torch.squeeze(torch.gather(p, -1, x0.unsqueeze(-1)), -1)
                x0_p[:, block_end:] = -np.inf

                # Frontier masking
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    bias = checker.compute_mask(vocab_size=logits_with_noise.shape[-1])
                    logits_with_noise[0, consume_idx, bias] = -np.inf
                    x0[0, consume_idx] = torch.argmax(logits_with_noise[0, consume_idx])

                x0         = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break
                batch_k = min(current_batch, n_scheduled - tokens_placed, n_available)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)
                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vi  = x0[0, pos].item()
                    if logits_with_noise[0, pos, vi] == -np.inf:
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    return x, resamples, coverage_records

                tokens_placed += len(positions)

                new_idx, violator = extend_prefix(checker, x, consume_idx, mask_id)

                if violator < 0:
                    consume_idx   = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                else:
                    consume_idx = new_idx

                    if checker.is_accepting():
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    # ── PROBE: measure top-k coverage at violator position ──
                    pos_logits = logits[0, violator]          # (vocab,)
                    valid_bias = checker.compute_mask(vocab_size=pos_logits.shape[0])
                    # valid_bias is True where BLOCKED, False where allowed
                    allowed_mask = ~valid_bias
                    n_allowed = allowed_mask.sum().item()

                    if n_allowed > 0:
                        valid_logits = pos_logits.clone()
                        valid_logits[valid_bias] = -torch.inf
                        valid_probs = F.softmax(valid_logits.float(), dim=-1)

                        sorted_probs, _ = valid_probs.sort(descending=True)
                        cumsum = sorted_probs.cumsum(0)

                        rec = {"n_allowed": n_allowed, "position": violator - gen_start}
                        for k in [1, 5, 10, 20, 50, 100, 200]:
                            idx_k = min(k, n_allowed) - 1
                            rec[f"top{k}_coverage"] = cumsum[idx_k].item()
                        coverage_records.append(rec)

                    bad_token = x[0, violator].item()
                    logits_with_noise[0, violator, bad_token] = -np.inf
                    x[0, violator] = mask_id
                    tokens_placed -= 1
                    resamples.append(violator)

                    if len(resamples) >= max_resamples:
                        return x, resamples, coverage_records

                    found = False
                    while len(resamples) < max_resamples:
                        nv = torch.argmax(logits_with_noise[0, violator]).item()
                        if logits_with_noise[0, violator, nv] == -np.inf:
                            break
                        c = checker.matcher.try_consume_tokens([nv])
                        if c == 1:
                            x[0, violator] = nv
                            consume_idx  += 1
                            tokens_placed += 1
                            found = True
                            fi, fv = extend_prefix(checker, x, consume_idx, mask_id)
                            consume_idx = fi
                            break
                        logits_with_noise[0, violator, nv] = -np.inf
                        resamples.append(violator)
                    current_batch = 1

                if not complete and checker.is_accepting():
                    gen_ids    = x[0, gen_start:].tolist()
                    first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids))
                    if first_mask >= consume_idx - gen_start:
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

    return x, resamples, coverage_records


def run_exp_a(instances, eval_model, tokenizer, model, seed, steps, out_file):
    print(f"[Exp A] top-k coverage probe, {len(instances)} instances")
    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            continue
        try:
            checker = TokenChecker(schema_str)
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: {e}")
            continue

        prompt_ids, prompt_len, suffix, start_line, _ = eval_model.prepare_prompt(
            instance, tokenizer, model, trace=False
        )
        torch.manual_seed(seed)
        t0  = time.monotonic()
        out, resamples, coverage = _generate_with_coverage_probe(
            model, prompt_ids, checker, prompt_len,
            steps=steps, gen_length=256, block_length=32, temperature=0.2,
        )
        elapsed = time.monotonic() - t0
        valid   = _is_valid(out, prompt_ids.shape[1])

        # Aggregate coverage stats
        agg = {}
        if coverage:
            for k in [1, 5, 10, 20, 50, 100, 200]:
                vals = [r[f"top{k}_coverage"] for r in coverage]
                agg[f"top{k}_mean"] = sum(vals) / len(vals)
                agg[f"top{k}_p5"]   = sorted(vals)[max(0, int(len(vals) * 0.05) - 1)]
                agg[f"top{k}_p95"]  = sorted(vals)[int(len(vals) * 0.95)]

        result = {
            "instance_id": instance.instance_id(),
            "experiment": "A_topk_coverage",
            "valid": valid,
            "time_taken": elapsed,
            "n_violator_positions": len(coverage),
            "coverage_agg": agg,
            "coverage_records": coverage,
        }
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "a") as f:
            print(json.dumps(result), file=f, flush=True)

        top50 = agg.get("top50_mean", float("nan"))
        print(f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
              f"valid={valid}, time={elapsed:.1f}s, "
              f"n_violators={len(coverage)}, top50_mean={top50:.3f}")


# ── Experiment B: batch size ablation ────────────────────────────────────────

@torch.no_grad()
def _generate_fixed_batch(
    model, prompt, checker, prompt_len,
    steps=128, gen_length=256, block_length=32,
    temperature=0.2, max_batch_size=1, max_resamples=100,
):
    """Generate with a fixed max_batch_size (no AIMD growth above this cap)."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks      = gen_length // block_length
    steps_per_block = steps // num_blocks
    gen_start       = prompt.shape[1]
    consume_idx     = gen_start
    current_batch   = 1
    resamples       = []
    total_violations = 0

    if prompt_len < gen_start:
        checker.consume_tokens(x[0, prompt_len:gen_start].tolist())

    fwd_times = []

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end   = gen_start + (num_block + 1) * block_length
        block_mask_index   = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            t_fwd = time.perf_counter()
            logits            = model(x).logits
            fwd_times.append((time.perf_counter() - t_fwd) * 1000)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            n_scheduled       = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed = 0
            while tokens_placed < n_scheduled:
                if complete:
                    break

                mask_index = x == mask_id
                x0         = torch.argmax(logits_with_noise, dim=-1)
                p          = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p       = torch.squeeze(torch.gather(p, -1, x0.unsqueeze(-1)), -1)
                x0_p[:, block_end:] = -np.inf

                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    bias = checker.compute_mask(vocab_size=logits_with_noise.shape[-1])
                    logits_with_noise[0, consume_idx, bias] = -np.inf
                    x0[0, consume_idx] = torch.argmax(logits_with_noise[0, consume_idx])

                x0         = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break
                batch_k = min(current_batch, n_scheduled - tokens_placed, n_available)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)
                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vi  = x0[0, pos].item()
                    if logits_with_noise[0, pos, vi] == -np.inf:
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    return x, resamples, total_violations, fwd_times

                tokens_placed += len(positions)
                new_idx, violator = extend_prefix(checker, x, consume_idx, mask_id)

                if violator < 0:
                    consume_idx   = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                else:
                    total_violations += 1
                    consume_idx = new_idx

                    if checker.is_accepting():
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    bad_token = x[0, violator].item()
                    logits_with_noise[0, violator, bad_token] = -np.inf
                    x[0, violator] = mask_id
                    tokens_placed -= 1
                    resamples.append(violator)

                    if len(resamples) >= max_resamples:
                        return x, resamples, total_violations, fwd_times

                    while len(resamples) < max_resamples:
                        nv = torch.argmax(logits_with_noise[0, violator]).item()
                        if logits_with_noise[0, violator, nv] == -np.inf:
                            break
                        c = checker.matcher.try_consume_tokens([nv])
                        if c == 1:
                            x[0, violator] = nv
                            consume_idx  += 1
                            tokens_placed += 1
                            fi, _ = extend_prefix(checker, x, consume_idx, mask_id)
                            consume_idx = fi
                            break
                        logits_with_noise[0, violator, nv] = -np.inf
                        resamples.append(violator)
                    current_batch = 1

                if not complete and checker.is_accepting():
                    gen_ids    = x[0, gen_start:].tolist()
                    first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids))
                    if first_mask >= consume_idx - gen_start:
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

    return x, resamples, total_violations, fwd_times


def run_exp_b(instances, eval_model, tokenizer, model, seed, steps, out_file,
              batch_sizes=(1, 2, 4, 8)):
    print(f"[Exp B] batch size ablation {list(batch_sizes)}, {len(instances)} instances")
    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            continue

        prompt_ids, prompt_len, suffix, start_line, _ = eval_model.prepare_prompt(
            instance, tokenizer, model, trace=False
        )
        gen_start = prompt_ids.shape[1]

        row = {"instance_id": instance.instance_id(), "experiment": "B_batch_ablation",
               "results": {}}

        for bs in batch_sizes:
            try:
                checker = TokenChecker(schema_str)
            except Exception as e:
                row["results"][str(bs)] = {"error": str(e)}
                continue

            torch.manual_seed(seed)
            t0 = time.monotonic()
            out, resamples, violations, fwd_times = _generate_fixed_batch(
                model, prompt_ids, checker, prompt_len,
                steps=steps, gen_length=256, block_length=32,
                temperature=0.2, max_batch_size=bs,
            )
            elapsed = time.monotonic() - t0
            valid   = _is_valid(out, gen_start)
            fwd_mean = sum(fwd_times) / len(fwd_times) if fwd_times else 0

            row["results"][str(bs)] = {
                "valid": valid,
                "time_taken": elapsed,
                "resamples": len(resamples),
                "violations": violations,
                "forward_mean_ms": fwd_mean,
                "forward_count": len(fwd_times),
            }
            print(f"  [{i+1}/{len(instances)}] {instance.instance_id()} bs={bs}: "
                  f"valid={valid}, time={elapsed:.1f}s, "
                  f"resamples={len(resamples)}, violations={violations}")

        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "a") as f:
            print(json.dumps(row), file=f, flush=True)


# ── Experiment D: validity comparison (dgrammar vs unconstrained) ─────────────

@torch.no_grad()
def _generate_unconstrained(model, prompt, steps=128, gen_length=256,
                             block_length=32, temperature=0.2):
    """Plain masked diffusion with no grammar checking."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks      = gen_length // block_length
    steps_per_block = steps // num_blocks
    gen_start       = prompt.shape[1]

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end   = gen_start + (num_block + 1) * block_length
        block_mask_index   = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            logits            = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            n_scheduled       = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            mask_index = x == mask_id
            x0         = torch.argmax(logits_with_noise, dim=-1)
            p          = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p       = torch.squeeze(torch.gather(p, -1, x0.unsqueeze(-1)), -1)
            x0_p[:, block_end:] = -np.inf

            x0         = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

            n_available = mask_index[0].sum().item()
            if n_available == 0:
                break
            batch_k = min(n_scheduled, n_available)
            _, select_indices = torch.topk(confidence[0], k=batch_k)
            for idx in select_indices:
                pos = idx.item()
                x[0, pos] = x0[0, pos]

    return x


def run_exp_d(instances, eval_model, tokenizer, model, seed, steps, out_file):
    print(f"[Exp D] dgrammar vs unconstrained, {len(instances)} instances")
    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            continue

        prompt_ids, prompt_len, suffix, start_line, _ = eval_model.prepare_prompt(
            instance, tokenizer, model, trace=False
        )
        gen_start = prompt_ids.shape[1]

        # --- dgrammar ---
        try:
            checker = TokenChecker(schema_str)
            torch.manual_seed(seed)
            t0 = time.monotonic()
            out_dg, rs_dg, _, _ = _generate_fixed_batch(
                model, prompt_ids, checker, prompt_len,
                steps=steps, gen_length=256, block_length=32,
                temperature=0.2, max_batch_size=8,
            )
            t_dg    = time.monotonic() - t0
            valid_dg = _is_valid(out_dg, gen_start)
        except Exception as e:
            valid_dg, t_dg, rs_dg = False, 0.0, []
            print(f"  dgrammar error {instance.instance_id()}: {e}")

        # --- unconstrained ---
        torch.manual_seed(seed)
        t0 = time.monotonic()
        out_un   = _generate_unconstrained(model, prompt_ids, steps=steps,
                                           gen_length=256, block_length=32, temperature=0.2)
        t_un     = time.monotonic() - t0
        valid_un = _is_valid(out_un, gen_start)

        result = {
            "instance_id": instance.instance_id(),
            "experiment": "D_validity_comparison",
            "dgrammar":      {"valid": valid_dg, "time_taken": t_dg, "resamples": len(rs_dg)},
            "unconstrained": {"valid": valid_un, "time_taken": t_un},
        }
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "a") as f:
            print(json.dumps(result), file=f, flush=True)

        print(f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
              f"dgrammar={valid_dg}({t_dg:.1f}s) unconstrained={valid_un}({t_un:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    exp          = sys.argv[1].upper() if len(sys.argv) > 1 else "C"
    seed         = int(sys.argv[2])    if len(sys.argv) > 2 else 0
    limit        = int(sys.argv[3])    if len(sys.argv) > 3 else 50
    dataset_name = sys.argv[4]         if len(sys.argv) > 4 else "jsb_medium"
    steps        = int(sys.argv[5])    if len(sys.argv) > 5 else 128
    offset       = int(sys.argv[6])    if len(sys.argv) > 6 else 0

    ds_safe  = dataset_name.replace("/", "_")
    sfx      = f"_off{offset}" if offset > 0 else ""
    out_file = f"results/exp_{exp}_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"

    dataset, eval_model, tokenizer, model = _load_resources(dataset_name)
    torch.manual_seed(seed)

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances     = all_instances[offset: offset + limit]

    print(f"Experiment {exp}: {len(instances)} instances, seed={seed}, T={steps}, out={out_file}")

    if exp == "A":
        run_exp_a(instances, eval_model, tokenizer, model, seed, steps, out_file)

    elif exp == "B":
        batch_sizes_arg = sys.argv[7] if len(sys.argv) > 7 else "1,2,4,8"
        batch_sizes = tuple(int(x) for x in batch_sizes_arg.split(","))
        run_exp_b(instances, eval_model, tokenizer, model, seed, steps, out_file, batch_sizes)

    elif exp == "C":
        # Exp C reuses run_dgrammar_timed.py directly — just call it via subprocess
        import subprocess, os
        subprocess.run(
            ["python", "/root/run_dgrammar_timed.py",
             str(seed), str(limit), dataset_name, str(steps), str(offset), "1"],
            cwd="/root",
            env={**os.environ, "PYTHONPATH": "/root:/root/CD4dLLM"},
        )

    elif exp == "D":
        run_exp_d(instances, eval_model, tokenizer, model, seed, steps, out_file)

    else:
        print(f"Unknown experiment: {exp}. Choose A, B, C, or D.")
        sys.exit(1)


if __name__ == "__main__":
    main()

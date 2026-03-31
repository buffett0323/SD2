"""Run LAVE (CD4dLLM 'our' method) with per-operation timing on jsonschema.

LAVE uses llguidance Checker with beam search validation and AR fallback.
We monkey-patch the Checker to record per-operation timing.

Dataset registry name (argv[3]): e.g. ``jsonschema`` (json-mode-eval) or
``jsonschemabench`` (``epfl-dlab/JSONSchemaBench``). For JSONSchemaBench README
metrics (syntactic/functional/time/constraint %), each JSONL line includes
``schema`` so you can run ``bench/jsonschemabench_metrics.py`` on the output.
"""

import json
import time
import sys
from pathlib import Path

import torch

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model


class LAVETimingStats:
    """Collect per-operation timing for LAVE's Checker calls."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []       # model forward pass times (seconds)
        self.validate_times = []      # validate_tokens calls (seconds)
        self.consume_times = []       # consume_tokens calls (seconds)
        self.compute_mask_times = []  # compute_logit_bias calls (seconds)
        self.rollback_times = []      # rollback calls (seconds)
        self.retry_count = 0

    def summary(self):
        fwd = self.forward_times
        val = self.validate_times
        con = self.consume_times
        cm = self.compute_mask_times
        rb = self.rollback_times
        return {
            "forward_count": len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms": (sum(fwd) / len(fwd) * 1000) if fwd else 0,
            "validate_count": len(val),
            "validate_total_ms": sum(val) * 1000,
            "validate_mean_ms": (sum(val) / len(val) * 1000) if val else 0,
            "consume_count": len(con),
            "consume_total_ms": sum(con) * 1000,
            "consume_mean_ms": (sum(con) / len(con) * 1000) if con else 0,
            "compute_mask_count": len(cm),
            "compute_mask_total_ms": sum(cm) * 1000,
            "compute_mask_mean_ms": (sum(cm) / len(cm) * 1000) if cm else 0,
            "rollback_count": len(rb),
            "rollback_total_ms": sum(rb) * 1000,
            "retry_count": self.retry_count,
        }


STATS = LAVETimingStats()


def patch_checker_class():
    """Monkey-patch the CD4dLLM Checker class to record timing."""
    from constrained_diffusion.checker_tokenizer import Checker

    _orig_validate = Checker.validate_tokens
    _orig_consume = Checker.consume_tokens
    _orig_compute_mask = Checker.compute_mask
    _orig_rollback = Checker.rollback

    def timed_validate(self, next_tokens):
        t0 = time.perf_counter()
        result = _orig_validate(self, next_tokens)
        STATS.validate_times.append(time.perf_counter() - t0)
        return result

    def timed_consume(self, next_tokens):
        t0 = time.perf_counter()
        result = _orig_consume(self, next_tokens)
        STATS.consume_times.append(time.perf_counter() - t0)
        return result

    def timed_compute_mask(self):
        t0 = time.perf_counter()
        result = _orig_compute_mask(self)
        STATS.compute_mask_times.append(time.perf_counter() - t0)
        return result

    def timed_rollback(self, count):
        t0 = time.perf_counter()
        result = _orig_rollback(self, count)
        STATS.rollback_times.append(time.perf_counter() - t0)
        return result

    Checker.validate_tokens = timed_validate
    Checker.consume_tokens = timed_consume
    Checker.compute_mask = timed_compute_mask
    Checker.rollback = timed_rollback


def patch_model_forward(model):
    """Wrap model forward to record timing."""
    _orig_forward = model.forward

    def timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_forward(*args, **kwargs)
        STATS.forward_times.append(time.perf_counter() - t0)
        return result

    model.forward = timed_forward
    return model


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    tag = "lave_timed"
    ds_safe = dataset_name.replace("/", "_")
    sfx = f"_off{offset}" if offset > 0 else ""
    output_file = f"results/{tag}_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"

    # Patch Checker before any imports that use it
    patch_checker_class()

    # Import LAVE's generate function
    from constrained_diffusion.eval.dllm.models.llada.generate_our import generate as lave_generate

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")
    model = patch_model_forward(model)

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset:offset + limit]
    print(f"LAVE timed: {len(instances)} instances, seed={seed}, T={steps}")

    for i, instance in enumerate(instances):
        # Get the per-instance lark grammar (LAVE uses lark, not JSON schema)
        try:
            cfg_lang = instance.cfg()
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: {e}")
            continue

        prompt_ids, input_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        STATS.reset()
        torch.manual_seed(seed)
        start_time = time.monotonic()

        try:
            out, total_retry_num, gen_start_time = lave_generate(
                model,
                tokenizer,
                prompt_ids,
                input_len=input_len,
                grammar=cfg_lang,
                steps=steps,
                gen_length=256,
                block_length=32,
                temperature=0.2,
                remasking="low_confidence",
                trace=False,
                change_logits=False,
                top_k_per_mask=5,
                top_n_beam=30,
                random_n_beam=20,
                max_retry_num_total=1000,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(instances)}] {instance.instance_id()}: ERROR {e}")
            continue

        elapsed = time.monotonic() - start_time
        STATS.retry_count = total_retry_num

        if out is None:
            code = "TIMEOUT"
            extracted = None
            valid = False
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]
            extracted = instance.extract_result(suffix + start_line + code)

            # Check validity: no mask tokens before EOS
            gen_ids = out[0, prompt_ids.shape[1]:].tolist()
            eos_id, eot_id, mask_id = 126081, 126348, 126336
            valid = False
            if eos_id in gen_ids or eot_id in gen_ids:
                eos_pos = next((j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None)
                valid = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

        timing = STATS.summary()

        # Constraint = everything except forward passes
        total_constraint_ms = (
            timing["validate_total_ms"]
            + timing["consume_total_ms"]
            + timing["compute_mask_total_ms"]
            + timing["rollback_total_ms"]
        )
        total_forward_ms = timing["forward_total_ms"]
        tokens = 256

        result = {
            "instance_id": instance.instance_id(),
            "method": "lave",
            "dataset": dataset_name,
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": total_retry_num,
            "timing": {
                **timing,
                "total_constraint_ms": total_constraint_ms,
                "total_forward_ms": total_forward_ms,
                "constraint_pct": (total_constraint_ms / (total_constraint_ms + total_forward_ms) * 100)
                    if (total_constraint_ms + total_forward_ms) > 0 else 0,
                "per_token_constraint_ms": total_constraint_ms / tokens,
                "per_token_total_ms": elapsed * 1000 / tokens,
            },
        }
        data = getattr(instance, "data", None)
        if isinstance(data, dict) and data.get("schema") is not None:
            result["schema"] = data["schema"]

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        fwd_mean = timing["forward_mean_ms"]
        val_mean = timing["validate_mean_ms"]
        con_mean = timing["consume_mean_ms"]
        cm_mean = timing["compute_mask_mean_ms"]
        cpct = result["timing"]["constraint_pct"]
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, time={elapsed:.1f}s, "
            f"fwd={fwd_mean:.0f}ms(x{timing['forward_count']}), "
            f"validate={val_mean:.2f}ms(x{timing['validate_count']}), "
            f"consume={con_mean:.2f}ms(x{timing['consume_count']}), "
            f"compute_mask={cm_mean:.2f}ms(x{timing['compute_mask_count']}), "
            f"retries={total_retry_num}, constraint={cpct:.1f}%"
        )


if __name__ == "__main__":
    main()

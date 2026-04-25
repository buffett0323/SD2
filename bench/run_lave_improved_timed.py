"""Run LAVE with one of four improvement directions (or all combined).

Usage:
    python run_lave_improved_timed.py <seed> <limit> <dataset> <steps> <offset> \
        <instance_timeout> <experiment>

experiment must be one of: dir1 | dir2 | dir3 | dir4 | combined

  dir1  - Block-level Joint Verification:
            Use block_length=128 so each denoising step unmasks a larger chunk,
            reducing the number of independent verification boundaries.

  dir2  - Deterministic Wildcard Verification:
            random_n_beam=0, top_n_beam=50  →  pure top-k lookahead with no
            random sampling, approaching 0% false-negative from lookahead.

  dir3  - Incremental Parsing Cache:
            Monkey-patch compute_mask so results are cached per consumed-prefix.
            When the same parser state is revisited (e.g. after rollback to an
            already-seen prefix), the mask is returned instantly.

  dir4  - Grammar-Guided Token Filtering:
            change_logits=True  →  LAVE filters the vocabulary via grammar
            *before* proposing tokens, like AR constrained decoding.

  combined - dir2 + dir3 + dir4 applied together (dir1 keeps its default
            block_length so the block structure stays comparable).
"""

import json
import time
import sys
import signal
from pathlib import Path

import torch


class InstanceTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise InstanceTimeout("Instance timeout")


from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
import jsb_dataset  # noqa: F401 - registers jsb_* datasets


# ── Timing stats ─────────────────────────────────────────────────────────────

class LAVETimingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.validate_times = []
        self.consume_times = []
        self.compute_mask_times = []
        self.rollback_times = []
        self.cache_hits = 0
        self.retry_count = 0

    def summary(self):
        fwd = self.forward_times
        val = self.validate_times
        con = self.consume_times
        cm  = self.compute_mask_times
        rb  = self.rollback_times
        return {
            "forward_count":           len(fwd),
            "forward_total_ms":        sum(fwd) * 1000,
            "forward_mean_ms":         (sum(fwd) / len(fwd) * 1000) if fwd else 0,
            "validate_count":          len(val),
            "validate_total_ms":       sum(val) * 1000,
            "validate_mean_ms":        (sum(val) / len(val) * 1000) if val else 0,
            "consume_count":           len(con),
            "consume_total_ms":        sum(con) * 1000,
            "consume_mean_ms":         (sum(con) / len(con) * 1000) if con else 0,
            "compute_mask_count":      len(cm),
            "compute_mask_total_ms":   sum(cm) * 1000,
            "compute_mask_mean_ms":    (sum(cm) / len(cm) * 1000) if cm else 0,
            "rollback_count":          len(rb),
            "rollback_total_ms":       sum(rb) * 1000,
            "cache_hits":              self.cache_hits,
            "retry_count":             self.retry_count,
        }


STATS = LAVETimingStats()


# ── Monkey-patches ────────────────────────────────────────────────────────────

def patch_checker_class(use_cache: bool = False):
    """Monkey-patch CD4dLLM Checker to record timing and optionally cache masks."""
    from constrained_diffusion.checker_tokenizer import Checker

    _orig_validate     = Checker.validate_tokens
    _orig_consume      = Checker.consume_tokens
    _orig_compute_mask = Checker.compute_mask
    _orig_rollback     = Checker.rollback

    # Per-instance state for dir3 cache (keyed by checker id)
    _prefix_buf: dict[int, list] = {}   # checker id → list of consumed token ids
    _mask_cache: dict[int, dict] = {}   # checker id → {prefix_key → mask}

    def timed_validate(self, next_tokens):
        t0 = time.perf_counter()
        result = _orig_validate(self, next_tokens)
        STATS.validate_times.append(time.perf_counter() - t0)
        return result

    def timed_consume(self, next_tokens):
        t0 = time.perf_counter()
        result = _orig_consume(self, next_tokens)
        STATS.consume_times.append(time.perf_counter() - t0)
        if use_cache:
            cid = id(self)
            if cid not in _prefix_buf:
                _prefix_buf[cid] = []
                _mask_cache[cid] = {}
            toks = list(next_tokens) if hasattr(next_tokens, "__iter__") else [next_tokens]
            _prefix_buf[cid].extend(toks)
        return result

    def timed_rollback(self, count):
        t0 = time.perf_counter()
        result = _orig_rollback(self, count)
        STATS.rollback_times.append(time.perf_counter() - t0)
        if use_cache:
            cid = id(self)
            if cid in _prefix_buf:
                del _prefix_buf[cid][-count:]
        return result

    if use_cache:
        def cached_compute_mask(self):
            cid = id(self)
            if cid not in _prefix_buf:
                _prefix_buf[cid] = []
                _mask_cache[cid] = {}
            key = tuple(_prefix_buf[cid])
            if key in _mask_cache[cid]:
                STATS.cache_hits += 1
                STATS.compute_mask_times.append(0.0)  # instant cache hit
                return _mask_cache[cid][key]
            t0 = time.perf_counter()
            result = _orig_compute_mask(self)
            STATS.compute_mask_times.append(time.perf_counter() - t0)
            _mask_cache[cid][key] = result
            return result

        Checker.compute_mask = cached_compute_mask
    else:
        def timed_compute_mask(self):
            t0 = time.perf_counter()
            result = _orig_compute_mask(self)
            STATS.compute_mask_times.append(time.perf_counter() - t0)
            return result

        Checker.compute_mask = timed_compute_mask

    Checker.validate_tokens = timed_validate
    Checker.consume_tokens  = timed_consume
    Checker.rollback        = timed_rollback


def patch_model_forward(model):
    _orig_forward = model.forward

    def timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        result = _orig_forward(*args, **kwargs)
        STATS.forward_times.append(time.perf_counter() - t0)
        return result

    model.forward = timed_forward
    return model


# ── Experiment parameter tables ───────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    # dir1: larger blocks → each denoising step covers a wider joint window,
    #   reducing per-block verification calls from 256/32=8 down to 256/128=2.
    "dir1": dict(
        block_length=128,
        change_logits=False,
        top_k_per_mask=5,
        top_n_beam=30,
        random_n_beam=20,
        use_cache=False,
    ),
    # dir2: deterministic-only beam search, more top-k beams to compensate.
    "dir2": dict(
        block_length=32,
        change_logits=False,
        top_k_per_mask=5,
        top_n_beam=50,
        random_n_beam=0,
        use_cache=False,
    ),
    # dir3: incremental parsing cache on compute_mask.
    "dir3": dict(
        block_length=32,
        change_logits=False,
        top_k_per_mask=5,
        top_n_beam=30,
        random_n_beam=20,
        use_cache=True,
    ),
    # dir4: grammar-guided token filtering before sampling.
    "dir4": dict(
        block_length=32,
        change_logits=True,
        top_k_per_mask=5,
        top_n_beam=30,
        random_n_beam=20,
        use_cache=False,
    ),
    # combined: dir2 + dir3 + dir4 (block_length stays at 32 for fair comparison).
    "combined": dict(
        block_length=32,
        change_logits=True,
        top_k_per_mask=5,
        top_n_beam=50,
        random_n_beam=0,
        use_cache=True,
    ),
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    seed             = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit            = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name     = sys.argv[3]      if len(sys.argv) > 3 else "jsonschema"
    steps            = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset           = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    instance_timeout = int(sys.argv[6]) if len(sys.argv) > 6 else 120
    experiment       = sys.argv[7]      if len(sys.argv) > 7 else "dir4"

    if experiment not in EXPERIMENT_CONFIGS:
        print(f"Unknown experiment '{experiment}'. Choose: {list(EXPERIMENT_CONFIGS)}")
        sys.exit(1)

    cfg = EXPERIMENT_CONFIGS[experiment]

    tag     = f"lave_{experiment}_timed"
    ds_safe = dataset_name.replace("/", "_")
    sfx     = f"_off{offset}" if offset > 0 else ""
    output_file = f"results/{tag}_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"

    # Patch Checker *before* importing generate
    patch_checker_class(use_cache=cfg["use_cache"])

    from constrained_diffusion.eval.dllm.models.llada.generate_our import generate as lave_generate

    dataset    = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model     = eval_model.model("cuda")
    model     = patch_model_forward(model)

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances     = all_instances[offset:offset + limit]
    print(
        f"LAVE [{experiment}]: {len(instances)} instances, seed={seed}, T={steps}, "
        f"block_length={cfg['block_length']}, change_logits={cfg['change_logits']}, "
        f"top_n_beam={cfg['top_n_beam']}, random_n_beam={cfg['random_n_beam']}, "
        f"cache={cfg['use_cache']}"
    )

    for i, instance in enumerate(instances):
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

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(instance_timeout)
        try:
            out, total_retry_num, gen_start_time = lave_generate(
                model,
                tokenizer,
                prompt_ids,
                input_len=input_len,
                grammar=cfg_lang,
                steps=steps,
                gen_length=256,
                block_length=cfg["block_length"],
                temperature=0.2,
                remasking="low_confidence",
                trace=False,
                change_logits=cfg["change_logits"],
                top_k_per_mask=cfg["top_k_per_mask"],
                top_n_beam=cfg["top_n_beam"],
                random_n_beam=cfg["random_n_beam"],
                max_retry_num_total=1000,
            )
        except InstanceTimeout:
            signal.alarm(0)
            elapsed = time.monotonic() - start_time
            print(
                f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
                f"TIMEOUT ({elapsed:.1f}s)"
            )
            result = {
                "instance_id": instance.instance_id(),
                "method": f"lave_{experiment}",
                "experiment": experiment,
                "valid": False,
                "extracted": None,
                "time_taken": elapsed,
                "resamples": 0,
                "timing": {"timeout": True},
            }
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "a") as f:
                print(json.dumps(result), flush=True, file=f)
            continue
        except Exception as e:
            signal.alarm(0)
            elapsed = time.monotonic() - start_time
            print(f"  [{i+1}/{len(instances)}] {instance.instance_id()}: ERROR {e}")
            result = {
                "instance_id": instance.instance_id(),
                "method": f"lave_{experiment}",
                "experiment": experiment,
                "experiment_config": {
                    k: v for k, v in cfg.items() if k != "use_cache"
                },
                "valid": False,
                "extracted": None,
                "time_taken": elapsed,
                "resamples": 0,
                "timing": {"error": True, "message": str(e)},
            }
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "a") as f:
                print(json.dumps(result), flush=True, file=f)
            continue
        signal.alarm(0)

        elapsed = time.monotonic() - start_time
        STATS.retry_count = total_retry_num

        if out is None:
            extracted = None
            valid = False
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]
            extracted = instance.extract_result(suffix + start_line + code)

            gen_ids = out[0, prompt_ids.shape[1]:].tolist()
            eos_id, eot_id, mask_id = 126081, 126348, 126336
            valid = False
            if eos_id in gen_ids or eot_id in gen_ids:
                eos_pos = next(
                    (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
                )
                valid = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

        timing = STATS.summary()

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
            "method": f"lave_{experiment}",
            "experiment": experiment,
            "experiment_config": {
                k: v for k, v in cfg.items() if k != "use_cache"
            },
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": total_retry_num,
            "timing": {
                **timing,
                "total_constraint_ms": total_constraint_ms,
                "total_forward_ms": total_forward_ms,
                "constraint_pct": (
                    total_constraint_ms / (total_constraint_ms + total_forward_ms) * 100
                ) if (total_constraint_ms + total_forward_ms) > 0 else 0,
                "per_token_constraint_ms": total_constraint_ms / tokens,
                "per_token_total_ms": elapsed * 1000 / tokens,
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        fwd_mean = timing["forward_mean_ms"]
        cm_mean  = timing["compute_mask_mean_ms"]
        cpct     = result["timing"]["constraint_pct"]
        hits     = timing["cache_hits"]
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, time={elapsed:.1f}s, "
            f"fwd={fwd_mean:.0f}ms(x{timing['forward_count']}), "
            f"compute_mask={cm_mean:.2f}ms(x{timing['compute_mask_count']}), "
            f"cache_hits={hits}, retries={total_retry_num}, "
            f"constraint={cpct:.1f}%"
        )


if __name__ == "__main__":
    main()

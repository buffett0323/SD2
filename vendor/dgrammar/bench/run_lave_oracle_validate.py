"""Line-2 generation: oracle-guided ``validate()`` (replace beam failure when possible).

When LAVE's ``validate()`` returns False, run ``oracle_find_block_assignment`` (DFS over
the block). If a concrete token assignment exists, write it into ``all_token_ids`` for
that block and call the original ``validate()`` again. Side effects on ``checker`` and
``cache_seq`` are handled entirely inside ``validate()`` — we only rewrite MASK positions
to a grammar-consistent assignment.

This is **not** the observational FN bench: it changes generation outcomes (fewer retries
when the oracle rescues a block that beam missed).

Usage: same argv as ``run_lave_fn_detection.py`` (seed, limit, dataset, steps, offset,
instance_timeout, oracle_mask_limit, optional oracle_max_search_seconds, oracle_sample_rate).

Env: same oracle env vars as ``oracle_fast`` / FN bench (``DGRAMMAR_ORACLE_*``).
``DGRAMMAR_ORACLE_SEARCH_MODE``: ``dfs`` (single assignment DFS) or ``smart`` (dedup + DFS
unsat short-circuit, then assignment DFS when possibly sat). Modal oracle-validate defaults
to ``smart``.

Output JSONL: ``results/lave_oracle_validate_<dataset>_s<seed>_t<steps>_oml<oml>...jsonl``
with ``line2_summary`` per finished instance.
"""

import json
import os
import random
import sys
import time
import signal
from pathlib import Path
from typing import Optional

import torch

ORACLE_VALIDATE_SCRIPT_VERSION = 1

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from oracle_fast import (
    init_oracle,
    get_oracle_state,
    get_oracle_mask_limit,
    oracle_find_block_assignment,
)


class InstanceTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise InstanceTimeout("Instance timeout")


from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
import constrained_diffusion.eval.dllm.models.llada.generate_our as _gour
import jsb_dataset  # noqa: F401

mask_id = 126336
eos_id = 126081
eot_id = 126348


def _oracle_max_search_seconds_resolved() -> Optional[float]:
    if len(sys.argv) > 8:
        v = float(sys.argv[8])
        return None if v <= 0 else v
    raw = os.environ.get("DGRAMMAR_ORACLE_MAX_SEARCH_SECONDS", "").strip()
    if raw:
        v = float(raw)
        return None if v <= 0 else v
    return 3.0


def _oracle_sample_rate_resolved() -> float:
    if len(sys.argv) > 9:
        v = float(sys.argv[9])
        return min(1.0, max(0.0, v))
    raw = os.environ.get("DGRAMMAR_ORACLE_SAMPLE_RATE", "").strip()
    if raw:
        v = float(raw)
        return min(1.0, max(0.0, v))
    return 1.0


_LINE2_COUNTERS = {
    "oracle_rescues": 0,
    "second_validate_failures": 0,
    "rejects_no_assignment": 0,
    "skipped_no_masks": 0,
    "skipped_high_masks": 0,
    "oracle_disabled_budget": 0,
}

_LINE2_DEBUG = os.environ.get("DGRAMMAR_LINE2_DEBUG", "").strip().lower() in ("1", "true", "yes")


def patch_validate_oracle_replace(
    grammar_str_getter,
    oracle_mask_limit: int,
    oracle_max_search_seconds: Optional[float] = None,
    oracle_sample_rate: float = 1.0,
    oracle_call_mask_limit: Optional[int] = None,
    oracle_no_rescue_budget: Optional[int] = None,
):
    import constrained_diffusion.eval.dllm.models.llada.generate_our as gen_mod

    _orig_validate = gen_mod.validate
    # Per-instance state: consecutive no-rescue oracle calls.
    _no_rescue_streak = [0]
    _oracle_disabled = [False]

    def intercepted_validate(
        checker,
        all_token_ids,
        p,
        index_to_consume,
        last_token_index,
        min_eos_eot_index,
        trace=False,
        top_k_per_mask=10,
        top_n_beam=30,
        random_n_beam=20,
    ):
        # Snapshot prefix BEFORE _orig_validate, which may advance checker.tokens.
        prefix_tokens_snapshot = list(checker.tokens)

        result = _orig_validate(
            checker,
            all_token_ids,
            p,
            index_to_consume,
            last_token_index,
            min_eos_eot_index,
            trace,
            top_k_per_mask,
            top_n_beam,
            random_n_beam,
        )
        if result:
            return True

        grammar_str = grammar_str_getter()
        if grammar_str is None:
            return False

        if last_token_index < index_to_consume:
            return False

        block_slice = slice(index_to_consume, last_token_index + 1)
        block_snapshot = list(all_token_ids[block_slice])
        block_tokens = list(block_snapshot)

        # Early exit: no MASKs in block → oracle cannot help, avoid expensive DFS.
        n_masks = sum(1 for t in block_tokens if t == mask_id)
        if _LINE2_DEBUG:
            print(
                f"[line2] n_masks={n_masks}, block_len={len(block_tokens)}, "
                f"index_to_consume={index_to_consume}",
                flush=True,
            )
        if n_masks == 0:
            _LINE2_COUNTERS["skipped_no_masks"] += 1
            return False

        # Skip if block has too many MASKs: DFS is exponential and nearly always TN.
        # oracle_call_mask_limit is a tighter per-call gate than oracle_mask_limit
        # (which is checked inside oracle_find_block_assignment but after prefix sync).
        if oracle_call_mask_limit is not None and n_masks > oracle_call_mask_limit:
            _LINE2_COUNTERS["skipped_high_masks"] += 1
            return False

        # Per-instance budget: if oracle has failed N consecutive times without a
        # rescue, disable it for the rest of this instance. This prevents oracle
        # overhead from accumulating on high-retry instances (like o10518/o10617)
        # where LAVE can self-recover with many fast retries.
        if _oracle_disabled[0]:
            _LINE2_COUNTERS["oracle_disabled_budget"] += 1
            return False

        ost = get_oracle_state()
        if ost is None:
            return False

        if oracle_sample_rate < 1.0 and random.random() >= oracle_sample_rate:
            return False

        # Use the pre-validate prefix snapshot so oracle's checker syncs correctly.
        prefix_tokens = prefix_tokens_snapshot

        timing: dict = {}
        assignment = oracle_find_block_assignment(
            ost,
            prefix_tokens,
            block_tokens,
            get_oracle_mask_limit(),
            timing_out=timing,
            max_search_seconds=oracle_max_search_seconds,
            model_logits=p,
            logits_index_base=index_to_consume,
        )

        if assignment is None:
            _LINE2_COUNTERS["rejects_no_assignment"] += 1
            _no_rescue_streak[0] += 1
            if oracle_no_rescue_budget is not None and _no_rescue_streak[0] >= oracle_no_rescue_budget:
                _oracle_disabled[0] = True
            return False

        if len(assignment) != len(block_tokens):
            _LINE2_COUNTERS["rejects_no_assignment"] += 1
            _no_rescue_streak[0] += 1
            if oracle_no_rescue_budget is not None and _no_rescue_streak[0] >= oracle_no_rescue_budget:
                _oracle_disabled[0] = True
            return False

        for i, tok in enumerate(assignment):
            all_token_ids[index_to_consume + i] = tok

        result2 = _orig_validate(
            checker,
            all_token_ids,
            p,
            index_to_consume,
            last_token_index,
            min_eos_eot_index,
            trace,
            top_k_per_mask,
            top_n_beam,
            random_n_beam,
        )

        if not result2:
            _LINE2_COUNTERS["second_validate_failures"] += 1
            _no_rescue_streak[0] += 1
            if oracle_no_rescue_budget is not None and _no_rescue_streak[0] >= oracle_no_rescue_budget:
                _oracle_disabled[0] = True
            for i, tok in enumerate(block_snapshot):
                all_token_ids[index_to_consume + i] = tok
            return False

        # Successful rescue — reset streak.
        _no_rescue_streak[0] = 0
        _LINE2_COUNTERS["oracle_rescues"] += 1
        return True

    def reset_instance_state() -> None:
        _no_rescue_streak[0] = 0
        _oracle_disabled[0] = False

    gen_mod.validate = intercepted_validate
    return _orig_validate, reset_instance_state


class TimingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []

    def summary(self):
        fwd = self.forward_times
        return {
            "forward_count": len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms": (sum(fwd) / len(fwd) * 1000) if fwd else 0,
        }


STATS = TimingStats()


def patch_model_forward(model):
    _orig = model.forward

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        r = _orig(*args, **kwargs)
        STATS.forward_times.append(time.perf_counter() - t0)
        return r

    model.forward = timed
    return model


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    instance_timeout = int(sys.argv[6]) if len(sys.argv) > 6 else 120
    oracle_mask_limit = int(sys.argv[7]) if len(sys.argv) > 7 else 12
    oracle_max_search_seconds = _oracle_max_search_seconds_resolved()
    oracle_sample_rate = _oracle_sample_rate_resolved()
    oracle_call_mask_limit: Optional[int] = None
    if len(sys.argv) > 10:
        v = int(sys.argv[10])
        oracle_call_mask_limit = v if v > 0 else None
    oracle_no_rescue_budget: Optional[int] = None
    if len(sys.argv) > 11:
        v = int(sys.argv[11])
        oracle_no_rescue_budget = v if v > 0 else None

    ds_safe = dataset_name.replace("/", "_")
    sfx = f"_off{offset}" if offset > 0 else ""
    out_file = (
        f"results/lave_oracle_validate_{ds_safe}_s{seed}_t{steps}_oml{oracle_mask_limit}{sfx}.jsonl"
    )

    _current_grammar = {"str": None}
    _, reset_oracle_instance = patch_validate_oracle_replace(
        lambda: _current_grammar["str"],
        oracle_mask_limit,
        oracle_max_search_seconds,
        oracle_sample_rate,
        oracle_call_mask_limit,
        oracle_no_rescue_budget,
    )

    from constrained_diffusion.eval.dllm.models.llada.generate_our import generate as lave_generate

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")
    model = patch_model_forward(model)

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset : offset + limit]
    print(
        f"LAVE oracle-validate (line2): {len(instances)} instances, seed={seed}, T={steps}, "
        f"oracle_mask_limit={oracle_mask_limit}, oracle_max_search_seconds={oracle_max_search_seconds!r}"
    )

    for i, instance in enumerate(instances):
        try:
            cfg_lang = instance.cfg()
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: cfg() error: {e}")
            continue

        _current_grammar["str"] = cfg_lang
        for k in _LINE2_COUNTERS:
            _LINE2_COUNTERS[k] = 0
        reset_oracle_instance()
        random.seed(seed + i)
        init_oracle(cfg_lang, oracle_mask_limit)

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
        except InstanceTimeout:
            signal.alarm(0)
            elapsed = time.monotonic() - start_time
            retries_at_timeout = _gour.last_total_retry_num
            print(
                f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
                f"TIMEOUT ({elapsed:.1f}s), retries={retries_at_timeout}"
            )
            result = {
                "instance_id": instance.instance_id(),
                "oracle_validate_script_version": ORACLE_VALIDATE_SCRIPT_VERSION,
                "valid": False,
                "extracted": None,
                "time_taken": elapsed,
                "resamples": retries_at_timeout,
                "line2_summary": dict(_LINE2_COUNTERS),
                "timing": {"timeout": True},
            }
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "a") as f:
                print(json.dumps(result), flush=True, file=f)
            continue
        except Exception as e:
            signal.alarm(0)
            elapsed = time.monotonic() - start_time
            print(f"  [{i+1}/{len(instances)}] {instance.instance_id()}: ERROR {e}")
            result = {
                "instance_id": instance.instance_id(),
                "oracle_validate_script_version": ORACLE_VALIDATE_SCRIPT_VERSION,
                "valid": False,
                "extracted": None,
                "time_taken": elapsed,
                "resamples": _gour.last_total_retry_num,
                "line2_summary": dict(_LINE2_COUNTERS),
                "timing": {"error": True, "message": str(e)},
            }
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "a") as f:
                print(json.dumps(result), flush=True, file=f)
            continue
        signal.alarm(0)

        elapsed = time.monotonic() - start_time

        if out is None:
            extracted, valid = None, False
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1] :], skip_special_tokens=True
            )[0]
            extracted = instance.extract_result(suffix + start_line + code)
            gen_ids = out[0, prompt_ids.shape[1] :].tolist()
            valid = False
            if eos_id in gen_ids or eot_id in gen_ids:
                eos_pos = next(
                    (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
                )
                valid = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

        result = {
            "instance_id": instance.instance_id(),
            "oracle_validate_script_version": ORACLE_VALIDATE_SCRIPT_VERSION,
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": total_retry_num,
            "line2_summary": dict(_LINE2_COUNTERS),
            "timing": {
                **STATS.summary(),
                "per_token_total_ms": elapsed * 1000 / 256,
            },
        }

        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        ls = _LINE2_COUNTERS
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, retries={total_retry_num}, time={elapsed:.1f}s, "
            f"rescues={ls['oracle_rescues']}, no_assign={ls['rejects_no_assignment']}, "
            f"second_val_fail={ls['second_validate_failures']}, "
            f"skip_no_masks={ls['skipped_no_masks']}, skip_high_masks={ls['skipped_high_masks']}, "
            f"disabled={ls['oracle_disabled_budget']}"
        )


if __name__ == "__main__":
    main()

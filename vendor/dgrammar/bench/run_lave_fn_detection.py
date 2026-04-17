"""Detect LAVE false negatives using an exhaustive oracle check.

When LAVE's beam-search validate() returns False (rejecting a block), we
immediately run an oracle that searches only over the **block** tokens (not the
full prefix): ``oracle_fast`` syncs a cached ``Checker`` to the current prefix
incrementally, then searches the block (default: ``dfs`` — sound). Enumeration uses
``compute_mask()`` for MASK positions. ``bfs_dedup`` / ``smart`` (phase-1) use a
heuristic merge key (``UserWarning`` once); ``smart`` then runs sound ``dfs`` when
phase-1 is False.

Comparing modes (dfs / bfs / bfs_dedup / smart):
    Run the same CLI args twice with different env, write to different files
    (or rename outputs between runs), then compare ``fn_summary`` / per-event
    ``oracle_ms`` in the JSONL. Example::

        DGRAMMAR_ORACLE_SEARCH_MODE=dfs python run_lave_fn_detection.py ...
        DGRAMMAR_ORACLE_SEARCH_MODE=bfs_dedup python run_lave_fn_detection.py ...

    For a fair timing comparison, fix ``seed``, ``limit``, ``dataset``, ``steps``,
    and ``offset``. ``bfs_dedup`` uses a heuristic merge; if ``dfs`` vs
    ``bfs_dedup`` disagree on any block, treat that as a fingerprint collision
    worth reporting.

If the oracle returns True while LAVE returned False → false negative **logged**
(``type: false_negative``). The patched ``validate`` still **returns False** so
LAVE retries as usual — we never replace a reject with ``True`` (that would
corrupt generation and cause timeouts / ``resamples=0``).

This script is OBSERVATIONAL: we only append to ``fn_events``; outcome for LAVE
is unchanged from the unpatched validator.

Usage:
    python run_lave_fn_detection.py <seed> <limit> <dataset> <steps> <offset> \\
        <instance_timeout> <oracle_mask_limit> [<oracle_max_search_seconds> [<oracle_sample_rate>]]

    oracle_mask_limit: max MASK tokens in a block before skipping oracle (default 12)
    oracle_max_search_seconds: cap on block search time after prefix sync (default 3.0; 0 = no cap)
    oracle_sample_rate: fraction of rejects on which to run the oracle (default 1.0)

    Env: DGRAMMAR_ORACLE_SEARCH_MODE=dfs|bfs|bfs_dedup|smart
         (default dfs; bfs_dedup/smart use a heuristic dedup key; optional
         DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS= id,id,... refines the key; smart
         confirms unsat with DFS when phase-1 dedup is False)

    Env: DGRAMMAR_ORACLE_MAX_SEARCH_SECONDS=S (optional; also argv[8])
         Per-call cap on **block search** time after prefix sync. Default in this script: 3.0.
         Use 0 to disable. Prevents instance wall-clock blowups on heavy rejects.

    Env: DGRAMMAR_ORACLE_SAMPLE_RATE=p (optional; also argv[9])
         On each LAVE reject, run the oracle with probability ``p`` in [0, 1]; default 1.0.
         Use e.g. 0.1 to cut oracle overhead ~10× while estimating FN rate (higher variance).
         ``fn_rate`` is over sampled FN+TN events only (not extrapolated to unsampled rejects).

    Env: DGRAMMAR_ORACLE_LOGITS_PRIORITY=0|1 (optional; default 1)
         When 1, oracle DFS/BFS tries MASK candidates in descending model ``p`` score
         (same tensor as ``validate()``). Set 0 to use grammar mask iteration order.

    Env: DGRAMMAR_ORACLE_TRIE_ORDER=0|1 (optional; default 0)
         When 1, order MASK candidates by trie DFS over ``LLTokenizer.dbg_tokens`` strings
         (prefix grouping). With logits on, logits sort first, trie breaks ties.

    Env: DGRAMMAR_ORACLE_STRING_PRUNE_THRESHOLD=N (optional)
         If MASK has >N allowed tokens, try only one — fast but **unsound** vs full DFS.

    Each FN/TN event includes ``oracle_impl: "oracle_fast"`` and split timings when
    ``oracle_verify_fast`` ran successfully (``oracle_prefix_sync_ms``,
    ``oracle_search_ms``). Rows from older runs without these keys used a different
    code path or predated timing — re-run to compare. ``oracle_ms`` equals
    ``timing_out["total_ms"]`` for that call (≈ prefix + search + tiny overhead).

Output (JSONL):
    Each line = one instance result, with extra "fn_events" list recording every
    block where LAVE said False but oracle said True.
    ``resamples`` matches LAVE's ``total_retry_num`` when generation finishes; on
    timeout or error it is read from ``generate_our.last_total_retry_num`` (not 0).
"""

import json
import os
import random
import sys
import time
import signal
from pathlib import Path
from typing import List, Optional

import torch

# Bump when observational semantics or oracle wiring change (visible in JSONL).
FN_DETECTION_SCRIPT_VERSION = 7  # 7 = optional prefix trie order (DGRAMMAR_ORACLE_TRIE_ORDER) + timings

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from oracle_fast import (
    init_oracle,
    get_oracle_state,
    get_oracle_mask_limit,
    oracle_verify_fast,
)


class InstanceTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise InstanceTimeout("Instance timeout")


from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
import constrained_diffusion.eval.dllm.models.llada.generate_our as _gour
import jsb_dataset  # noqa: F401 – registers jsb_* datasets


mask_id = 126336
eos_id  = 126081
eot_id  = 126348


def _oracle_max_search_seconds_resolved() -> Optional[float]:
    """argv[8] overrides env; default 3.0s. <=0 disables cap."""
    if len(sys.argv) > 8:
        v = float(sys.argv[8])
        return None if v <= 0 else v
    raw = os.environ.get("DGRAMMAR_ORACLE_MAX_SEARCH_SECONDS", "").strip()
    if raw:
        v = float(raw)
        return None if v <= 0 else v
    return 3.0


def _oracle_sample_rate_resolved() -> float:
    """argv[9] or env ``DGRAMMAR_ORACLE_SAMPLE_RATE``; default 1.0. Clamped to [0, 1]."""
    if len(sys.argv) > 9:
        v = float(sys.argv[9])
        return min(1.0, max(0.0, v))
    raw = os.environ.get("DGRAMMAR_ORACLE_SAMPLE_RATE", "").strip()
    if raw:
        v = float(raw)
        return min(1.0, max(0.0, v))
    return 1.0


def _mask_bucket(n_masks: int) -> str:
    """Bucket for oracle timing analysis (matches common reporting bands)."""
    if n_masks <= 3:
        return "0-3"
    if n_masks <= 7:
        return "4-7"
    return "8-12"

# ── Oracle: incremental ``oracle_fast`` (see ``oracle_fast.py``) ──────────────

# ── Patch ─────────────────────────────────────────────────────────────────────

# Global mutable state shared between the patch and main()
_GRAMMAR_STR   = None          # set per-instance
_FN_EVENTS: List[dict] = []   # accumulated per-instance
_ORACLE_COUNTERS = {"oracle_sample_skips": 0}  # rejects where oracle was not sampled


def patch_validate(
    grammar_str_getter,
    oracle_mask_limit: int,
    oracle_max_search_seconds: Optional[float] = None,
    oracle_sample_rate: float = 1.0,
):
    """
    Monkey-patch generate_our.validate() to run the oracle when LAVE rejects.

    grammar_str_getter: callable that returns the current grammar string.
    Uses ``oracle_fast.oracle_verify_fast`` with per-instance ``init_oracle`` state.
    Always returns the same boolean as the original ``validate`` (observational).
    """
    import constrained_diffusion.eval.dllm.models.llada.generate_our as gen_mod

    _orig_validate = gen_mod.validate

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
        before_len = len(checker.tokens)
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
        after_len = len(checker.tokens)
        if after_len != before_len:
            print(
                f"[fn_detection] validate() changed checker.tokens len "
                f"{before_len} → {after_len} (result={result})"
            )

        if not result:
            grammar_str = grammar_str_getter()
            if grammar_str is None:
                return result

            prefix_tokens = list(checker.tokens)
            block_tokens = all_token_ids[index_to_consume : last_token_index + 1]

            if last_token_index < index_to_consume:
                return result

            ost = get_oracle_state()
            if ost is None:
                # init_oracle() was not called for this instance — oracle skipped
                return result

            if oracle_sample_rate < 1.0 and random.random() >= oracle_sample_rate:
                _ORACLE_COUNTERS["oracle_sample_skips"] += 1
                return result

            oracle_timing: dict = {}
            oracle_result = oracle_verify_fast(
                ost,
                prefix_tokens,
                block_tokens,
                get_oracle_mask_limit(),
                timing_out=oracle_timing,
                max_search_seconds=oracle_max_search_seconds,
                model_logits=p,
                logits_index_base=index_to_consume,
            )
            oracle_ms = float(oracle_timing.get("total_ms", 0.0))

            n_masks = sum(1 for t in block_tokens if t == mask_id)
            bucket = _mask_bucket(n_masks)

            def _oracle_event_base() -> dict:
                ps = float(oracle_timing.get("prefix_sync_ms", 0.0))
                ss = float(oracle_timing.get("search_ms", 0.0))
                ev = {
                    "oracle_impl": "oracle_fast",
                    "block_len": len(block_tokens),
                    "n_masks": n_masks,
                    "mask_bucket": bucket,
                    "high_mask_load": n_masks >= 8,
                    "prefix_len": len(prefix_tokens),
                    "oracle_ms": round(oracle_ms, 2),
                    "oracle_prefix_sync_ms": round(ps, 3),
                    "oracle_search_ms": round(ss, 3),
                    "oracle_search_mode": oracle_timing.get("search_mode"),
                    "oracle_string_prune_threshold": oracle_timing.get(
                        "string_prune_threshold"
                    ),
                    "oracle_logits_priority": oracle_timing.get("logits_priority"),
                    "oracle_logits_priority_active": oracle_timing.get(
                        "logits_priority_active"
                    ),
                    "oracle_trie_order": oracle_timing.get("trie_order"),
                    "oracle_trie_order_active": oracle_timing.get("trie_order_active"),
                }
                # Drop Nones only for optional string fields
                return {k: v for k, v in ev.items() if v is not None}

            if oracle_result is True:
                event = {"type": "false_negative", **_oracle_event_base()}
                _FN_EVENTS.append(event)
            elif oracle_result is False:
                event = {"type": "true_negative", **_oracle_event_base()}
                _FN_EVENTS.append(event)
            elif oracle_timing.get("skip_reason") == "oracle_search_timeout":
                event = {
                    "type": "skipped",
                    "skip_reason": "oracle_search_timeout",
                    **_oracle_event_base(),
                }
                _FN_EVENTS.append(event)

        return result

    gen_mod.validate = intercepted_validate
    return _orig_validate


# ── Timing ────────────────────────────────────────────────────────────────────

class TimingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.retry_count   = 0

    def summary(self):
        fwd = self.forward_times
        return {
            "forward_count":    len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms":  (sum(fwd) / len(fwd) * 1000) if fwd else 0,
        }


STATS = TimingStats()


def patch_model_forward(model):
    _orig = model.forward
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        r  = _orig(*args, **kwargs)
        STATS.forward_times.append(time.perf_counter() - t0)
        return r
    model.forward = timed
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    seed             = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit            = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name     = sys.argv[3]      if len(sys.argv) > 3 else "jsonschema"
    steps            = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset           = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    instance_timeout = int(sys.argv[6]) if len(sys.argv) > 6 else 120
    oracle_mask_limit= int(sys.argv[7]) if len(sys.argv) > 7 else 12
    oracle_max_search_seconds = _oracle_max_search_seconds_resolved()
    oracle_sample_rate = _oracle_sample_rate_resolved()

    ds_safe = dataset_name.replace("/", "_")
    sfx     = f"_off{offset}" if offset > 0 else ""
    out_file = f"results/lave_fn_detection_{ds_safe}_s{seed}_t{steps}_oml{oracle_mask_limit}{sfx}.jsonl"

    # ── Current grammar holder (mutated per instance) ─────────────────────────
    _current_grammar = {"str": None}
    patch_validate(
        lambda: _current_grammar["str"],
        oracle_mask_limit,
        oracle_max_search_seconds,
        oracle_sample_rate,
    )

    from constrained_diffusion.eval.dllm.models.llada.generate_our import generate as lave_generate

    dataset    = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model     = eval_model.model("cuda")
    model     = patch_model_forward(model)

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances     = all_instances[offset : offset + limit]
    print(
        f"LAVE FN detection: {len(instances)} instances, seed={seed}, T={steps}, "
        f"oracle_mask_limit={oracle_mask_limit}, "
        f"oracle_max_search_seconds={oracle_max_search_seconds!r}, "
        f"oracle_sample_rate={oracle_sample_rate}"
    )
    print(
        "  oracle: oracle_fast.OracleState + oracle_verify_fast "
        "(incremental prefix sync; per-event oracle_prefix_sync_ms / oracle_search_ms)"
    )

    for i, instance in enumerate(instances):
        try:
            cfg_lang = instance.cfg()
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: cfg() error: {e}")
            continue

        # Make grammar string available to the patch; one cached Checker per instance
        _current_grammar["str"] = cfg_lang
        _FN_EVENTS.clear()
        _ORACLE_COUNTERS["oracle_sample_skips"] = 0
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
                "fn_detection_script_version": FN_DETECTION_SCRIPT_VERSION,
                "valid": False,
                "extracted": None,
                "time_taken": elapsed,
                "resamples": retries_at_timeout,
                "fn_events": list(_FN_EVENTS),
                "fn_summary": {
                    "oracle_sample_rate": oracle_sample_rate,
                    "oracle_sample_skips": _ORACLE_COUNTERS["oracle_sample_skips"],
                },
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
                "fn_detection_script_version": FN_DETECTION_SCRIPT_VERSION,
                "valid": False,
                "extracted": None,
                "time_taken": elapsed,
                "resamples": _gour.last_total_retry_num,
                "fn_events": list(_FN_EVENTS),
                "fn_summary": {
                    "oracle_sample_rate": oracle_sample_rate,
                    "oracle_sample_skips": _ORACLE_COUNTERS["oracle_sample_skips"],
                },
                "timing": {"error": True, "message": str(e)},
            }
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "a") as f:
                print(json.dumps(result), flush=True, file=f)
            continue
        signal.alarm(0)

        elapsed = time.monotonic() - start_time
        STATS.retry_count = total_retry_num

        if out is None:
            extracted, valid = None, False
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]
            extracted = instance.extract_result(suffix + start_line + code)
            gen_ids   = out[0, prompt_ids.shape[1]:].tolist()
            valid     = False
            if eos_id in gen_ids or eot_id in gen_ids:
                eos_pos = next(
                    (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
                )
                valid = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

        # ── Summarise FN events ───────────────────────────────────────────────
        fn_events = list(_FN_EVENTS)
        n_fn   = sum(1 for e in fn_events if e["type"] == "false_negative")
        n_tn   = sum(1 for e in fn_events if e["type"] == "true_negative")
        n_skip = sum(1 for e in fn_events if e.get("type") == "skipped")
        fn_rate = n_fn / (n_fn + n_tn) if (n_fn + n_tn) > 0 else 0.0

        result = {
            "instance_id": instance.instance_id(),
            "fn_detection_script_version": FN_DETECTION_SCRIPT_VERSION,
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": total_retry_num,
            "fn_summary": {
                "false_negatives": n_fn,
                "true_negatives":  n_tn,
                "skipped_blocks":  n_skip,
                "fn_rate":         round(fn_rate, 4),
                "total_rejects":   n_fn + n_tn,
                "oracle_sample_rate": oracle_sample_rate,
                "oracle_sample_skips": _ORACLE_COUNTERS["oracle_sample_skips"],
            },
            "fn_events": fn_events,
            "timing": {
                **STATS.summary(),
                "per_token_total_ms": elapsed * 1000 / 256,
            },
        }

        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, retries={total_retry_num}, time={elapsed:.1f}s, "
            f"fn={n_fn}/{n_fn+n_tn} (rate={fn_rate:.1%}), skipped={n_skip}, "
            f"oracle_sample_skips={_ORACLE_COUNTERS['oracle_sample_skips']}"
        )


if __name__ == "__main__":
    main()

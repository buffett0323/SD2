"""Run IG-CD (ETH baseline) with per-operation timing instrumentation.

Same driver shape and JSONL conventions as ``run_lave_timed.py`` (``seed``, ``limit``,
``dataset``, ``steps``, ``offset`` / ``-off`` suffix), but constrained generation uses
CFG∩DFA (rustformlang) instead of llguidance / LAVE.

Dataset registry name (argv[3]): e.g. ``jsonschema`` or ``jsonschemabench``. Lines include
``dataset`` and ``schema`` (when the instance exposes it) for ``jsonschemabench_metrics.py``
and unified aggregation.

``valid`` matches ``run_lave_timed`` (no mask tokens before first EOS/EOT in the
generated span). ``grammar_complete`` is the IG-CD generator flag (EOS + no holes).

Records:
  - model_forward_ms: time per model(x).logits call
  - grammar_check_ms: time per CFG ∩ DFA emptiness check
  - token_select_ms: time per argmax + confidence computation
  - autocompletion_ms: time for autocompletion post-processing
  - total constraint overhead vs model forward breakdown
"""

import json
import time
import sys
from pathlib import Path

import torch

from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    preprocessed_generate_stuff,
    EOS,
    autocomplete_valid,
    partial_output_from_tokens,
    generated_language,
    derive_supertokens,
)
from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from rustformlang.cfg import CFG, is_intersection_empty_threaded


# --- Timing instrumentation ---

class TimingStats:
    """Collect per-operation timing."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.grammar_check_times = []
        self.token_select_times = []
        self.resample_count = 0
        self.tokens_unmasked = 0

    def summary(self):
        fwd = self.forward_times
        gc = self.grammar_check_times
        ts = self.token_select_times
        return {
            "forward_count": len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms": (sum(fwd) / len(fwd) * 1000) if fwd else 0,
            "grammar_check_count": len(gc),
            "grammar_check_total_ms": sum(gc) * 1000,
            "grammar_check_mean_ms": (sum(gc) / len(gc) * 1000) if gc else 0,
            "grammar_check_median_ms": (sorted(gc)[len(gc) // 2] * 1000) if gc else 0,
            "grammar_check_p95_ms": (sorted(gc)[int(len(gc) * 0.95)] * 1000) if len(gc) >= 20 else 0,
            "token_select_count": len(ts),
            "token_select_total_ms": sum(ts) * 1000,
            "token_select_mean_ms": (sum(ts) / len(ts) * 1000) if ts else 0,
            "resample_count": self.resample_count,
            "tokens_unmasked": self.tokens_unmasked,
        }


STATS = TimingStats()

# LLaDA-8B-Instruct special token ids (same as ``run_lave_timed.py``)
_EOS_ID, _EOT_ID, _MASK_ID = 126081, 126348, 126336
_GEN_LEN = 256


def diffusion_sequence_valid(out: torch.Tensor | None, prompt_ids: torch.Tensor) -> bool:
    """True iff first EOS/EOT lies in the gen span and no mask appears before it."""
    if out is None:
        return False
    gen_ids = out[0, prompt_ids.shape[1] :].tolist()
    if _EOS_ID not in gen_ids and _EOT_ID not in gen_ids:
        return False
    eos_pos = next((j for j, t in enumerate(gen_ids) if t in (_EOS_ID, _EOT_ID)), None)
    return eos_pos is not None and _MASK_ID not in gen_ids[:eos_pos]


def _patch_model_forward(model):
    """Wrap model.__call__ to record forward pass timing."""
    original_forward = model.forward

    def timed_forward(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_forward(*args, **kwargs)
        STATS.forward_times.append(time.perf_counter() - t0)
        return result

    model.forward = timed_forward
    return model


def _timed_check_valid(generated_words, constraint_lang, lex_map, terminals,
                       trace=False, prelex=None, subtokens={}, supertokens={},
                       strip_chars=None):
    """Wrapper around check_valid that records timing."""
    t0 = time.perf_counter()
    gen_lang = generated_language(
        generated_words, lex_map, terminals,
        trace=trace, prelex=prelex, subtokens=subtokens,
        supertokens=supertokens, strip_chars=strip_chars,
    )
    intersection_empty = is_intersection_empty_threaded(
        constraint_lang, gen_lang, timeout=100
    )
    STATS.grammar_check_times.append(time.perf_counter() - t0)
    return intersection_empty


# --- Patched generate function (from ETH's generate_constrained.py) ---

from constrained_diffusion.eval.dllm.models.llada.generate_constrained import (
    add_gumbel_noise,
    get_num_transfer_tokens,
)
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def generate_timed(
    model, prompt, tokenizer,
    constraint_lang, lex_map, prompt_len,
    steps=128, gen_length=256, block_length=32,
    temperature=0.0, remasking="low_confidence",
    mask_id=126336, prelex=None, subtokens={},
    strip_chars=None, additional_stuff=None,
    constrain=True, max_resamples=100,
):
    """IG-CD generate with per-operation timing."""
    if additional_stuff is None and constrain:
        additional_stuff = preprocessed_generate_stuff(
            tokenizer, constraint_lang, lex_map,
            prelex=prelex, subtokens=subtokens, strip_chars=strip_chars,
        )
    elif additional_stuff is None:
        additional_stuff = None, None, {}

    all_possible_lexings, no_lexing_tokens, supertokens = additional_stuff
    resamples = []

    if constrain:
        terminals = constraint_lang.get_terminals()
    else:
        terminals = None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    generated_words = tokenizer.batch_decode(x.squeeze())
    mask_decoded = tokenizer.decode(mask_id)
    generated_words = [w if w != mask_decoded else None for w in generated_words]
    eos_token = tokenizer.special_tokens_map["eos_token"]

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length:
               prompt.shape[1] + (num_block + 1) * block_length]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            # --- Model forward (timed via patch) ---
            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            for j in range(num_transfer_tokens[0, i]):
                if complete:
                    break
                mask_index = x == mask_id

                token_found = False
                while not token_found:
                    # --- Token selection (timed) ---
                    t_sel = time.perf_counter()
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )
                    elif remasking == "random":
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                    x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    _, select_index = torch.topk(confidence[0], k=1)
                    STATS.token_select_times.append(time.perf_counter() - t_sel)

                    if select_index.shape[0] == 0:
                        yield x, resamples, False
                        return

                    index_of_new_word = select_index.item()
                    new_word_vocab_index = x0[0][index_of_new_word]

                    if logits_with_noise[0][index_of_new_word][new_word_vocab_index] == -np.inf:
                        yield x, resamples, False
                        return

                    new_word = tokenizer.decode(new_word_vocab_index)
                    if new_word in (eos_token, "<|eot_id|>"):
                        new_word = EOS

                    generated_words[index_of_new_word] = new_word

                    # --- Grammar check (timed) ---
                    if constrain:
                        intersection_empty = _timed_check_valid(
                            generated_words[prompt_len:],
                            constraint_lang, lex_map, terminals,
                            prelex=prelex, subtokens=subtokens,
                            supertokens=supertokens, strip_chars=strip_chars,
                        )
                    else:
                        intersection_empty = False

                    if intersection_empty:
                        STATS.resample_count += 1
                        resamples.append((index_of_new_word, time.monotonic()))
                        logits_with_noise[0][index_of_new_word][new_word_vocab_index] = -np.inf
                        generated_words[index_of_new_word] = None

                        if len(resamples) >= max_resamples:
                            yield x, resamples, False
                            return
                        continue

                    token_found = True
                    STATS.tokens_unmasked += 1
                    transfer_index[0, select_index] = True
                    if new_word is EOS:
                        transfer_index[0, select_index:] = True
                        x0[0, select_index:] = x0[0, select_index]
                    x[transfer_index] = x0[transfer_index]

                    if EOS in generated_words:
                        if None not in generated_words[:generated_words.index(EOS)]:
                            complete = True
                            break

                    yield x, resamples, False

    is_complete = (
        EOS in generated_words
        and None not in generated_words[:generated_words.index(EOS)]
    )
    yield x, resamples, is_complete


# --- Main runner ---

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    tag = f"igcd_timed"
    ds_safe = dataset_name.replace("/", "_")
    suffix = f"_off{offset}" if offset > 0 else ""
    output_file = f"results/{tag}_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")

    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")
    model = _patch_model_forward(model)

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset:offset + limit]
    print(f"IG-CD timed: {len(instances)} instances, seed={seed}, T={steps}")

    lang, lex_map, subtokens = None, None, None
    prelex = None
    additional_stuff = None
    orig_lex_map = None

    for i, instance in enumerate(instances):
        try:
            if lang is None or dataset.different_grammar_per_instance:
                lang, lex_map, subtokens = instance.language_lex_subtokens()
                orig_lex_map = lex_map.copy()
                lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
                if instance.strip_chars() is not None and "\n" not in instance.strip_chars():
                    lex_map["lexFence"] = r"\n?```"
                else:
                    lex_map["lexFence"] = "```"
                orig_lex_map["lexFence"] = lex_map["lexFence"]
                lang = lang.to_normal_form()
                lex_map = compile_lex_map(lex_map, subtokens=subtokens)
                additional_stuff = None
                prelex = instance.prelex()

            if additional_stuff is None:
                additional_stuff = preprocessed_generate_stuff(
                    tokenizer, lang, lex_map,
                    prelex=prelex, subtokens=subtokens,
                    strip_chars=instance.strip_chars(),
                )
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: {e}")
            continue

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        STATS.reset()
        torch.manual_seed(seed)
        start_time = time.monotonic()

        out = None
        resamples = []
        grammar_complete = False
        try:
            for out, resamples, grammar_complete in generate_timed(
                model, prompt_ids, tokenizer,
                constraint_lang=lang, lex_map=lex_map,
                prompt_len=prompt_len,
                steps=steps, gen_length=_GEN_LEN, block_length=32,
                temperature=0.2, remasking="low_confidence",
                prelex=prelex, subtokens=subtokens,
                strip_chars=instance.strip_chars(),
                additional_stuff=additional_stuff,
                constrain=True, max_resamples=100,
            ):
                pass
        except Exception as e:
            print(f"  [{i+1}/{len(instances)}] {instance.instance_id()}: ERROR {e}")
            continue

        elapsed = time.monotonic() - start_time

        if out is None:
            code = "TIMEOUT"
            extracted = None
            valid = False
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]
            extracted = instance.extract_result(suffix_str + start_line + code)
            valid = diffusion_sequence_valid(out, prompt_ids)

        # Autocompletion
        autocompletion = None
        ac_time = 0.0
        supertokens_map = derive_supertokens(subtokens) if subtokens else {}
        if out is not None and not grammar_complete:
            ac_start = time.perf_counter()
            mask_id = 126336
            mask_decoded = tokenizer.decode(mask_id)
            eos_token = tokenizer.special_tokens_map["eos_token"]
            generated_words = tokenizer.batch_decode(out.squeeze())
            generated_words = [
                None if x == mask_decoded
                else EOS if x in (eos_token, "<|eot_id|>", "<|endoftext|>")
                else x
                for x in generated_words[prompt_len:]
            ]
            partial_output, first_token_gap, last_token_eos_adj = (
                partial_output_from_tokens(generated_words, prelex)
            )
            valid_completion = autocomplete_valid(
                partial_output=partial_output,
                first_token_gap=first_token_gap,
                last_token_eos_adj=last_token_eos_adj,
                generated_lang=generated_language(
                    generated_words,
                    lex_map, lang.get_terminals(),
                    prelex=prelex, subtokens=subtokens,
                    supertokens=supertokens_map,
                    strip_chars=instance.strip_chars(),
                ),
                subtokens=subtokens,
                lex_map=orig_lex_map,
                constraint_lang=lang,
            )
            ac_time = (time.perf_counter() - ac_start) * 1000
            if valid_completion is not None:
                autocompletion = instance.extract_result(suffix_str + valid_completion)

        timing = STATS.summary()
        total_constraint_ms = timing["grammar_check_total_ms"] + timing["token_select_total_ms"]
        total_forward_ms = timing["forward_total_ms"]

        result = {
            "instance_id": instance.instance_id(),
            "method": "igcd",
            "dataset": dataset_name,
            "valid": valid,
            "grammar_complete": grammar_complete,
            "extracted": extracted,
            "autocompletion": autocompletion,
            "time_taken": elapsed,
            "resamples": len(resamples),
            "timing": {
                **timing,
                "autocompletion_ms": ac_time,
                "total_constraint_ms": total_constraint_ms,
                "total_forward_ms": total_forward_ms,
                "constraint_pct": (total_constraint_ms / (total_constraint_ms + total_forward_ms) * 100)
                    if (total_constraint_ms + total_forward_ms) > 0 else 0,
                "per_token_constraint_ms": total_constraint_ms / _GEN_LEN,
                "per_token_total_ms": elapsed * 1000 / _GEN_LEN,
            },
        }
        data = getattr(instance, "data", None)
        if isinstance(data, dict) and data.get("schema") is not None:
            result["schema"] = data["schema"]

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        gc_mean = timing["grammar_check_mean_ms"]
        fwd_mean = timing["forward_mean_ms"]
        pct = result["timing"]["constraint_pct"]
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, igcd_complete={grammar_complete}, time={elapsed:.1f}s, "
            f"fwd={fwd_mean:.0f}ms, gc={gc_mean:.1f}ms, "
            f"constraint={pct:.0f}%, resamples={len(resamples)}"
        )


if __name__ == "__main__":
    main()

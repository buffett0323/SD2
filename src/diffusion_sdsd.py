"""
Diffusion loop with SDSD constraint at frontier.

Same structure as dgrammar.generate, but at the frontier (and violator retry)
uses our DINGO/Herding instead of argmax. Enables SDSD constrained decoding
in the diffusion paradigm.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from dgrammar.checker import TokenChecker
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens, extend_prefix

from bidirectional_dingo import bidirectional_gap_dingo, dfa_run
from csr_dfa import CSRTransitionMatrix, build_csr_from_transition_dict
from sparse_dingo import sparse_dingo_dp
from herding import HerdingMomentumState, herding_single_constrained_step
from baseline_dingo import baseline_dingo_dp

# Cap gap length for bidirectional DP (|Q| × k × K per layer).
MAX_BIDI_GAP = 48

FrontierPicker = Callable[..., int]


def _make_frontier_csr(valid_tokens: list[int], vocab_size: int):
    """Build minimal CSR: state 0 -> state 1 for each valid token."""
    if not valid_tokens:
        return None
    transitions = {(0, t): 1 for t in valid_tokens}
    return build_csr_from_transition_dict(transitions, num_states=2, vocab_size=vocab_size)


def _logits_to_prob_list(logits_1d: torch.Tensor, vocab_size: int) -> list[float]:
    logits_1d = torch.nan_to_num(
        logits_1d.float() if hasattr(logits_1d, "float") else logits_1d,
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )
    probs = F.softmax(logits_1d, dim=-1)
    if probs.dim() > 1:
        probs = probs[0]
    prob_list = probs.tolist()
    if len(prob_list) < vocab_size:
        prob_list.extend([0.0] * (vocab_size - len(prob_list)))
    return prob_list


def _clear_bidi_gap_ctx(ctx: dict[str, Any] | None) -> None:
    if ctx is None:
        return
    ctx.pop("bidi_tokens", None)
    ctx.pop("bidi_positions", None)


def _bidirectional_frontier_pick(
    logits_1d: torch.Tensor,
    checker: TokenChecker,
    vocab_size: int,
    ctx: dict[str, Any],
) -> int:
    """
    Multi-mask gap Viterbi on JSON DFA + fixed right suffix; schema-check the
    proposed chunk, then return first token (approximate CFG via DFA + verify).

    On success, stores the full optimal gap in ``ctx`` (``bidi_tokens``,
    ``bidi_positions``) so the diffusion loop can commit every mask in the gap.
    """
    x = ctx["x"]
    focus_pos: int = ctx["focus_pos"]
    logits_seq: torch.Tensor = ctx["logits_seq"]
    gen_start: int = ctx["gen_start"]
    mask_id: int = ctx["mask_id"]
    csr: CSRTransitionMatrix = ctx["csr"]
    json_start_state: int = ctx["json_start_state"]
    live_states: set[int] = ctx["live_states"]

    seq_len = x.shape[1]
    if focus_pos >= seq_len or int(x[0, focus_pos].item()) != mask_id:
        _clear_bidi_gap_ctx(ctx)
        return _frontier_picker_base(logits_1d, checker, vocab_size, sparse_dingo_dp)

    k = 0
    p = focus_pos
    while p < seq_len and int(x[0, p].item()) == mask_id:
        k += 1
        p += 1
    k = min(k, MAX_BIDI_GAP)

    suffix: list[int] = []
    q = focus_pos + k
    while q < seq_len:
        tid = int(x[0, q].item())
        if tid == mask_id:
            break
        suffix.append(tid)
        q += 1

    prefix_tokens = x[0, gen_start:focus_pos].tolist()
    q_left = dfa_run(csr, json_start_state, prefix_tokens)
    if q_left is None:
        _clear_bidi_gap_ctx(ctx)
        return _frontier_picker_base(logits_1d, checker, vocab_size, sparse_dingo_dp)

    prob_vectors: list[list[float]] = []
    for j in range(k):
        pos = focus_pos + j
        if pos >= seq_len:
            break
        prob_vectors.append(_logits_to_prob_list(logits_seq[pos], vocab_size))

    if len(prob_vectors) != k:
        _clear_bidi_gap_ctx(ctx)
        return _frontier_picker_base(logits_1d, checker, vocab_size, sparse_dingo_dp)

    res = bidirectional_gap_dingo(csr, q_left, prob_vectors, suffix, live_states)
    if not res.success or not res.tokens:
        _clear_bidi_gap_ctx(ctx)
        return _frontier_picker_base(logits_1d, checker, vocab_size, sparse_dingo_dp)

    first = res.tokens[0]
    bias = checker.compute_mask(vocab_size=vocab_size)
    if first < bias.shape[0] and bool(bias[first].item()):
        _clear_bidi_gap_ctx(ctx)
        return _frontier_picker_base(logits_1d, checker, vocab_size, sparse_dingo_dp)

    ctx["bidi_tokens"] = res.tokens
    ctx["bidi_positions"] = list(range(focus_pos, focus_pos + len(res.tokens)))
    return first


def _frontier_picker_base(
    logits_1d: torch.Tensor,
    checker: TokenChecker,
    vocab_size: int,
    decode_fn,
) -> int:
    """Generic frontier picker: get valid tokens, run decode_fn, return best token."""
    bias = checker.compute_mask(vocab_size=vocab_size)
    if hasattr(logits_1d, "cpu"):
        logits_1d = logits_1d.cpu().float()
    # Keep mask/logits on the same device for in-place masking.
    bias = bias.to(logits_1d.device)
    valid_tokens = [t for t in range(min(vocab_size, len(bias))) if not bool(bias[t].item())]
    if not valid_tokens:
        return -1
    csr = _make_frontier_csr(valid_tokens, vocab_size)
    if csr is None:
        return -1
    # add_gumbel_noise can produce +/-inf (exp(logits)/noise). If fed directly to
    # softmax, this may create NaNs and collapse DINGO/Herding to fallback tokens.
    logits_1d = torch.nan_to_num(logits_1d, nan=-1e9, posinf=1e9, neginf=-1e9)
    # Critical: mask invalid tokens *before* softmax. Otherwise, one huge invalid
    # logit can push all valid probabilities to 0 (underflow), causing fallback.
    masked_logits = logits_1d.clone()
    if bias.shape[0] > masked_logits.shape[0]:
        bias = bias[: masked_logits.shape[0]]
    elif bias.shape[0] < masked_logits.shape[0]:
        pad = torch.ones(
            masked_logits.shape[0] - bias.shape[0],
            dtype=torch.bool,
            device=masked_logits.device,
        )
        bias = torch.cat([bias, pad])
    masked_logits[bias] = float("-inf")
    if torch.isneginf(masked_logits).all():
        return -1
    probs = F.softmax(masked_logits, dim=-1)
    if probs.dim() > 1:
        probs = probs[0]
    prob_list = probs.tolist()
    if len(prob_list) < vocab_size:
        prob_list.extend([0.0] * (vocab_size - len(prob_list)))
    result = decode_fn(csr, [prob_list], 0, {1}, block_length=1)
    if not result.success or not result.tokens:
        return valid_tokens[0] if valid_tokens else -1
    return result.tokens[0]


def _herding_persistent_frontier(
    logits_1d: torch.Tensor,
    checker: TokenChecker,
    vocab_size: int,
    state: HerdingMomentumState,
) -> int:
    """
    Frontier pick with Herding momentum persisted across diffusion steps.

    Same masking / softmax as other pickers; one herding update per call.
    """
    bias = checker.compute_mask(vocab_size=vocab_size)
    if hasattr(logits_1d, "cpu"):
        logits_1d = logits_1d.cpu().float()
    bias = bias.to(logits_1d.device)
    valid_tokens = [t for t in range(min(vocab_size, len(bias))) if not bool(bias[t].item())]
    if not valid_tokens:
        return -1
    csr = _make_frontier_csr(valid_tokens, vocab_size)
    if csr is None:
        return -1

    logits_1d = torch.nan_to_num(logits_1d, nan=-1e9, posinf=1e9, neginf=-1e9)
    masked_logits = logits_1d.clone()
    if bias.shape[0] > masked_logits.shape[0]:
        bias = bias[: masked_logits.shape[0]]
    elif bias.shape[0] < masked_logits.shape[0]:
        pad = torch.ones(
            masked_logits.shape[0] - bias.shape[0],
            dtype=torch.bool,
            device=masked_logits.device,
        )
        bias = torch.cat([bias, pad])
    masked_logits[bias] = float("-inf")
    if torch.isneginf(masked_logits).all():
        return -1
    probs = F.softmax(masked_logits, dim=-1)
    if probs.dim() > 1:
        probs = probs[0]
    prob_list = probs.tolist()
    if len(prob_list) < vocab_size:
        prob_list.extend([0.0] * (vocab_size - len(prob_list)))

    w = state.ensure_w(vocab_size)
    best_t, ok = herding_single_constrained_step(
        prob_list, csr, 0, w, state.prev_token, delta=0.0
    )
    if not ok or best_t is None:
        return valid_tokens[0] if valid_tokens else -1
    state.prev_token = best_t
    return best_t


def make_frontier_picker(method: str, herding_state: HerdingMomentumState | None = None):
    """Create frontier_picker for sdsd, ablation1, ablation2, ablation3, baseline, argmax, bidi.

    Pickers accept optional 4th argument ``ctx`` (dict) for bidirectional gap DP.
    """

    def _argmax(logits_1d, checker, vocab_size, ctx=None):
        """Dgrammar-style: mask invalid, argmax. Use for debugging (verify loop works)."""
        bias = checker.compute_mask(vocab_size=vocab_size)
        if hasattr(logits_1d, "device"):
            logits = logits_1d.clone().float()
        else:
            logits = torch.tensor(logits_1d, dtype=torch.float32)
        bias = bias.to(logits.device)
        if bias.shape[0] > logits.shape[0]:
            bias = bias[: logits.shape[0]]
        elif bias.shape[0] < logits.shape[0]:
            pad = torch.ones(logits.shape[0] - bias.shape[0], dtype=torch.bool, device=logits.device)
            bias = torch.cat([bias, pad])
        logits = logits.clone()
        logits[bias] = float("-inf")
        return int(logits.argmax().item())

    def _sparse_dingo(logits_1d, checker, vocab_size, ctx=None):
        return _frontier_picker_base(logits_1d, checker, vocab_size, sparse_dingo_dp)

    def _baseline(logits_1d, checker, vocab_size, ctx=None):
        bias = checker.compute_mask(vocab_size=vocab_size)
        if hasattr(logits_1d, "cpu"):
            logits_1d = logits_1d.cpu().float()
        bias = bias.to(logits_1d.device)
        valid_tokens = [t for t in range(min(vocab_size, len(bias))) if not bool(bias[t].item())]
        if not valid_tokens:
            return -1

        def trans_fn(q, t):
            return 1 if (q == 0 and t in valid_tokens) else None

        logits_1d = torch.nan_to_num(logits_1d, nan=-1e9, posinf=1e9, neginf=-1e9)
        masked_logits = logits_1d.clone()
        if bias.shape[0] > masked_logits.shape[0]:
            bias = bias[: masked_logits.shape[0]]
        elif bias.shape[0] < masked_logits.shape[0]:
            pad = torch.ones(
                masked_logits.shape[0] - bias.shape[0],
                dtype=torch.bool,
                device=masked_logits.device,
            )
            bias = torch.cat([bias, pad])
        masked_logits[bias] = float("-inf")
        if torch.isneginf(masked_logits).all():
            return -1
        probs = F.softmax(masked_logits, dim=-1)
        if probs.dim() > 1:
            probs = probs[0]
        prob_list = probs.tolist()
        if len(prob_list) < vocab_size:
            prob_list.extend([0.0] * (vocab_size - len(prob_list)))
        result = baseline_dingo_dp(2, vocab_size, trans_fn, [prob_list], 0, {1}, block_length=1)
        if not result.success or not result.tokens:
            return valid_tokens[0] if valid_tokens else -1
        return result.tokens[0]

    if method in ("argmax", "dgrammar"):
        return _argmax
    if method in ("sdsd", "ablation3"):
        return _sparse_dingo
    if method == "ablation1":
        return _sparse_dingo
    if method == "ablation2":
        hs = herding_state if herding_state is not None else HerdingMomentumState()

        def _herding_persistent(logits_1d, checker, vocab_size, ctx=None):
            return _herding_persistent_frontier(logits_1d, checker, vocab_size, hs)

        return _herding_persistent
    if method == "baseline":
        return _baseline
    if method == "bidi":
        def _bidi(logits_1d, checker, vocab_size, ctx=None):
            if ctx is None:
                return _sparse_dingo(logits_1d, checker, vocab_size, None)
            return _bidirectional_frontier_pick(
                logits_1d, checker, vocab_size, ctx
            )

        return _bidi
    return _sparse_dingo


def generate_diffusion_sdsd(
    model,
    prompt: torch.Tensor,
    tokenizer,
    checker: TokenChecker,
    prompt_len: int,
    frontier_picker: FrontierPicker,
    steps: int = 128,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.2,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    eos_id: int = 126081,
    eot_id: int = 126348,
    max_batch_size: int = 8,
    max_resamples: int = 100,
    bidi_csr: CSRTransitionMatrix | None = None,
    bidi_live_states: set[int] | None = None,
    bidi_json_start_state: int = 0,
):
    """
    Diffusion generation with SDSD constraint at frontier.

    frontier_picker(logits_1d, checker, vocab_size, ctx=None) -> token_id
    When ``bidi_csr`` is set, ``ctx`` is passed with x, focus_pos, logits_seq for
    bidirectional gap Viterbi (see ``make_frontier_picker("bidi")``).  The full
    gap is written to ``x0`` with boosted confidence and ``topk`` batch size at
    least the gap length (capped by ``max_batch_size``) so the Viterbi tokens
    commit together.
    """
    start_time = time.monotonic()
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    total_violations = 0
    total_remasks = 0
    total_grammar_checks = 0
    resamples = []
    current_batch = 1
    gen_start = prompt.shape[1]
    consume_idx = gen_start

    if prompt_len < gen_start:
        prefix_tokens = x[0, prompt_len:gen_start].tolist()
        if not checker.consume_tokens(prefix_tokens):
            pass  # warning only

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed_this_step = 0
            vocab_size = logits_with_noise.shape[-1]

            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

                mask_index = x == mask_id
                x0 = torch.argmax(logits_with_noise, dim=-1)
                bidi_ctx = None

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                x0_p[:, block_end:] = -np.inf

                # SDSD frontier: use our picker instead of argmax
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    if bidi_csr is not None and bidi_live_states is not None:
                        bidi_ctx = {
                            "x": x,
                            "focus_pos": consume_idx,
                            "logits_seq": logits_with_noise[0],
                            "gen_start": gen_start,
                            "mask_id": mask_id,
                            "csr": bidi_csr,
                            "json_start_state": bidi_json_start_state,
                            "live_states": bidi_live_states,
                        }
                    tok = frontier_picker(
                        logits_with_noise[0, consume_idx],
                        checker,
                        vocab_size,
                        bidi_ctx,
                    )
                    if tok < 0:
                        _clear_bidi_gap_ctx(bidi_ctx)
                        bias = checker.compute_mask(vocab_size=vocab_size)
                        logits_with_noise[0, consume_idx, bias] = -np.inf
                        tok = torch.argmax(logits_with_noise[0, consume_idx]).item()
                    x0[0, consume_idx] = tok
                    if bidi_ctx is not None and bidi_ctx.get("bidi_tokens"):
                        if mask_index[0, consume_idx]:
                            x0_p[0, consume_idx] = 1.0
                        focus = consume_idx
                        for rel_pos, gap_tok in enumerate(bidi_ctx["bidi_tokens"][1:], start=1):
                            abs_pos = focus + rel_pos
                            if abs_pos < x.shape[1] and mask_index[0, abs_pos]:
                                x0[0, abs_pos] = gap_tok
                                x0_p[0, abs_pos] = 1.0

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break

                remaining = n_scheduled - tokens_placed_this_step
                batch_k = min(current_batch, remaining, n_available)
                if bidi_ctx is not None and bidi_ctx.get("bidi_tokens"):
                    min_gap = len(bidi_ctx["bidi_tokens"])
                    # Unmask the whole bidi gap in one topk (not only current_batch).
                    batch_k = min(max(batch_k, min_gap), n_available, max_batch_size)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)

                if select_indices.shape[0] == 0:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                bidi_commit: set[int] = set()
                if bidi_ctx is not None:
                    bidi_commit = set(bidi_ctx.get("bidi_positions", []))

                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vocab_idx = x0[0, pos].item()
                    if (
                        logits_with_noise[0, pos, vocab_idx] == -np.inf
                        and pos not in bidi_commit
                    ):
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                tokens_placed_this_step += len(positions)

                total_grammar_checks += 1
                new_idx, violator = extend_prefix(checker, x, consume_idx, mask_id)

                if violator < 0:
                    consume_idx = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                    _clear_bidi_gap_ctx(bidi_ctx)
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
                    total_remasks += 1
                    tokens_placed_this_step -= 1
                    resamples.append((violator, time.monotonic() - start_time))

                    if len(resamples) >= max_resamples:
                        yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                        return

                    # SDSD violator retry: use our picker instead of argmax
                    found = False
                    while len(resamples) < max_resamples:
                        bidi_ctx = None
                        if bidi_csr is not None and bidi_live_states is not None:
                            bidi_ctx = {
                                "x": x,
                                "focus_pos": violator,
                                "logits_seq": logits_with_noise[0],
                                "gen_start": gen_start,
                                "mask_id": mask_id,
                                "csr": bidi_csr,
                                "json_start_state": bidi_json_start_state,
                                "live_states": bidi_live_states,
                            }
                        next_vocab = frontier_picker(
                            logits_with_noise[0, violator],
                            checker,
                            vocab_size,
                            bidi_ctx,
                        )
                        if next_vocab < 0 or logits_with_noise[0, violator, next_vocab] == -np.inf:
                            break

                        total_grammar_checks += 1
                        c = checker.matcher.try_consume_tokens([next_vocab])
                        if c == 1:
                            x[0, violator] = next_vocab
                            consume_idx += 1
                            tokens_placed_this_step += 1
                            found = True
                            further_idx, further_viol = extend_prefix(
                                checker, x, consume_idx, mask_id
                            )
                            consume_idx = further_idx
                            break

                        logits_with_noise[0, violator, next_vocab] = -np.inf
                        total_remasks += 1
                        resamples.append((violator, time.monotonic() - start_time))

                    current_batch = 1

                if not complete and checker.is_accepting():
                    gen_ids = x[0, gen_start:].tolist()
                    first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids))
                    if first_mask >= consume_idx - gen_start:
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

                if not complete:
                    gen_ids = x[0, gen_start:].tolist()
                    if eos_id in gen_ids or eot_id in gen_ids:
                        eos_pos = None
                        for j, tid in enumerate(gen_ids):
                            if tid in (eos_id, eot_id):
                                eos_pos = j
                                break
                        if eos_pos is not None and mask_id not in gen_ids[:eos_pos]:
                            for j in range(eos_pos, len(gen_ids)):
                                x[0, gen_start + j] = x[0, gen_start + eos_pos]
                            complete = True

            yield x, resamples, False, total_violations, total_remasks, total_grammar_checks

    gen_ids = x[0, gen_start:].tolist()
    is_complete = False
    if eos_id in gen_ids or eot_id in gen_ids:
        eos_pos = next(
            (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
        )
        is_complete = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

    yield x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks

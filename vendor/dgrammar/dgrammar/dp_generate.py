"""DP-based grammar-constrained generation for diffusion LMs.

Instead of greedily retrying one violation at a time (generate.py), this module
runs a Viterbi DP over grammar DFA states across all non-mask positions in the
sequence. Key properties:

  - State = grammar DFA node, identified by bytes(matcher.compute_logit_bias()).
    Two token paths that land in the same DFA state are merged (Viterbi-style),
    keeping only the higher log-prob path. This bounds active states by DFA size,
    not by vocab^k.

  - Per position: O(|states| * top_k) rollback/advance probes (no cloning),
    then O(|next_states|) deep_copy() calls (one clone per surviving DFA state).

  - Returns the globally optimal token assignment for the entire non-mask
    prefix segment, rather than fixing one violator at a time.

Typical DFA size for JSON-schema grammars: 20–200 states, so the DP is fast.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from dgrammar.checker import TokenChecker
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens


# ── Core DP ──────────────────────────────────────────────────────────────────


def dp_fix_prefix(
    matcher,
    x: torch.Tensor,
    start_pos: int,
    log_probs: torch.Tensor,
    mask_id: int,
    top_k: int = 50,
    max_positions: int = 64,
) -> Optional[list[tuple[int, int]]]:
    """Find the highest log-prob grammar-valid token assignment for all non-mask
    positions in x starting at start_pos.

    Args:
        matcher: LLMatcher at the grammar state just before start_pos.
                 Pass ``checker.matcher.deep_copy()`` — this matcher is mutated.
        x: Current token sequence, shape [1, seq_len].
        start_pos: First position to process (inclusive). Stop at first mask.
        log_probs: Log-softmax probabilities, shape [1, seq_len, vocab_size].
        mask_id: Token ID for the MASK token.
        top_k: Number of top-scoring tokens to explore at each position.

    Returns:
        List of (position, new_token_id) for positions where the optimal token
        differs from the current x[0, pos], or [] if no changes are needed,
        or None if no grammar-valid path exists.
    """
    NEG_INF = -math.inf

    # Collect contiguous non-mask positions starting at start_pos.
    seq_len = x.shape[1]
    positions: list[int] = []
    p = start_pos
    while p < seq_len and x[0, p].item() != mask_id:
        positions.append(p)
        p += 1

    if not positions:
        return []

    # Cap segment length to avoid O(|states|×top_k×seg_len) blowup on long segments.
    if len(positions) > max_positions:
        positions = positions[:max_positions]

    # ── DP initialisation ────────────────────────────────────────────────────
    # states: state_key → (matcher_clone, cumulative_score)
    # back:   (step_index, state_key) → (prev_state_key, chosen_token_id)
    #
    # state_key = bytes(compute_logit_bias()) is a proxy for the DFA node.
    # Two paths reaching the same DFA node are merged; only the best survives.

    init_key: bytes = bytes(matcher.compute_logit_bias())
    states: dict[bytes, tuple] = {init_key: (matcher, 0.0)}
    back: dict[tuple, tuple] = {}

    # ── DP loop ──────────────────────────────────────────────────────────────
    for step, pos in enumerate(positions):
        pos_lp = log_probs[0, pos]          # [vocab_size]
        k = min(top_k, pos_lp.shape[0])
        top_lp, top_ids = torch.topk(pos_lp, k=k)

        # Phase 1 — exploration via rollback (no cloning).
        # For each active (prev_state, candidate_token) pair:
        #   - try consuming the token
        #   - record new DFA state and score if it beats the current winner
        #   - rollback to restore prev_state
        winners: dict[bytes, tuple] = {}   # new_key → (prev_key, tok_id, new_score)

        for prev_key, (prev_m, prev_score) in states.items():
            for lp_t, id_t in zip(top_lp, top_ids):
                lp = lp_t.item()
                if lp == NEG_INF:
                    continue
                tid = id_t.item()

                consumed = prev_m.try_consume_tokens([tid])
                if consumed < 1:
                    # Token rejected by grammar; try_consume_tokens consumed 0.
                    prev_m.rollback(0)   # safe no-op via checker.rollback guard
                    continue

                new_score = prev_score + lp
                new_key = bytes(prev_m.compute_logit_bias())
                prev_m.rollback(1)   # restore prev_m for the next candidate

                if new_key not in winners or winners[new_key][2] < new_score:
                    winners[new_key] = (prev_key, tid, new_score)

        if not winners:
            return None   # grammar dead-end: no valid token exists at this position

        # Phase 2 — build next_states by cloning one winner matcher per DFA state.
        # Cloning is O(|winners|) ≤ O(DFA size), not O(top_k * |states|).
        next_states: dict[bytes, tuple] = {}
        for new_key, (prev_key, tok_id, new_score) in winners.items():
            prev_m, _ = states[prev_key]
            new_m = prev_m.deep_copy()
            consumed = new_m.try_consume_tokens([tok_id])
            assert consumed == 1, (
                f"Phase-2 replay failed for token {tok_id} at pos {pos}: "
                f"expected 1 consumed, got {consumed}"
            )
            next_states[new_key] = (new_m, new_score)
            back[(step, new_key)] = (prev_key, tok_id)

        states = next_states

    # ── Backtrack ─────────────────────────────────────────────────────────────
    if not states:
        return None

    best_key = max(states, key=lambda k: states[k][1])

    replacements: list[tuple[int, int]] = []
    cur_key = best_key
    for step in range(len(positions) - 1, -1, -1):
        prev_key, tok_id = back[(step, cur_key)]
        orig_tok = x[0, positions[step]].item()
        if tok_id != orig_tok:
            replacements.append((positions[step], tok_id))
        cur_key = prev_key

    replacements.reverse()
    return replacements


# ── Async mask helper ────────────────────────────────────────────────────────


def _compute_mask_async(checker, vocab_size):
    """Run compute_mask in a background thread.

    Returns (thread, result_holder) where result_holder is a two-element list
    [bias_tensor_or_None, compute_time_seconds].  Call thread.join() before
    reading result_holder.
    """
    result = [None, 0.0]

    def _run():
        t0 = time.perf_counter()
        result[0] = checker.compute_mask(vocab_size=vocab_size)
        result[1] = time.perf_counter() - t0

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result


# ── Generation loop ──────────────────────────────────────────────────────────


def _extend_prefix(checker, x, consume_idx, mask_id):
    """Consume contiguous non-mask tokens from consume_idx.

    Returns (new_consume_idx, violator_pos_or_-1). Mirrors extend_prefix() in
    generate.py but without the STATS dependency so it can live here.
    """
    tokens = []
    pos = consume_idx
    while pos < x.shape[1]:
        tid = x[0, pos].item()
        if tid == mask_id:
            break
        tokens.append(tid)
        pos += 1
    if not tokens:
        return consume_idx, -1
    count = checker.matcher.try_consume_tokens(tokens)
    if count == len(tokens):
        return consume_idx + count, -1
    return consume_idx + count, consume_idx + count   # violator at consume_idx+count


@torch.no_grad()
def generate_dp(
    model,
    prompt,
    tokenizer,
    checker: TokenChecker,
    prompt_len: int,
    steps: int = 128,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    eos_id: int = 126081,
    eot_id: int = 126348,
    trace: bool = False,
    max_batch_size: int = 8,
    max_resamples: int = 100,
    top_k_dp: int = 100,
    max_dp_secs: float = 300.0,
    stats=None,
):
    """Dgrammar with DP-based violation correction + async mask overlap.

    Structurally identical to generate_async_timed: same token placement,
    frontier masking, and inner scheduling loop. Two improvements over the
    original:

      1. Async mask overlap: compute_mask for the frontier token is kicked off
         in a background thread before the forward pass so its CPU cost is
         hidden behind GPU time (same technique as generate_async_timed).

      2. top_k_dp=100: DP explores the top-100 tokens per position instead of
         top-50, increasing the chance of finding a grammar-valid path with
         negligible extra cost (DP is rarely triggered, ~1.2×/sample).

    Falls back to remasking the violator when DP finds no valid path.

    Yields:
        (x, resamples, is_complete, total_violations, total_fixes, total_dp_calls, consume_idx)
    """
    start_time = time.monotonic()

    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    gen_start = prompt.shape[1]
    consume_idx = gen_start
    current_batch = 1

    if prompt_len < gen_start:
        prefix_tokens = x[0, prompt_len:gen_start].tolist()
        if not checker.consume_tokens(prefix_tokens) and trace:
            print("Warning: prompt suffix rejected by checker")

    total_violations = 0
    total_dp_calls = 0
    total_fixes = 0
    resamples = []

    # Pending async mask result: (thread, result_holder) or None.
    # Kicked off just before each forward pass; joined just after.
    pending_mask = None

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            # ── Skip steps with nothing to place (avoid wasted forward pass) ─
            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            # ── Async mask kick-off (before forward pass) ────────────────────
            # If the frontier is still a mask token and we don't have a pending
            # result, start computing the grammar mask now so it runs in
            # parallel with the GPU forward pass.
            mask_index_pre = x == mask_id
            if (
                pending_mask is None
                and consume_idx < x.shape[1]
                and mask_index_pre[0, consume_idx]
            ):
                vocab_size_hint = 126464  # corrected after logits are available
                pending_mask = _compute_mask_async(checker, vocab_size_hint)

            t_fwd = time.perf_counter()
            logits = model(x).logits
            if stats is not None:
                stats.forward_times.append(time.perf_counter() - t_fwd)
            log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            tokens_placed_this_step = 0
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break
                if time.monotonic() - start_time > max_dp_secs:
                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                    return

                # ── Token selection (same as generate_async_timed) ───────────
                mask_index = x == mask_id
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                x0_p[:, block_end:] = -np.inf

                # ── Frontier masking: guarantee first unfilled token is valid ─
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    actual_vocab = logits_with_noise.shape[-1]
                    if pending_mask is not None:
                        # Async result: join (usually already done) and use it.
                        t_wait = time.perf_counter()
                        thread, result_holder = pending_mask
                        thread.join()
                        wait_s = time.perf_counter() - t_wait
                        pending_mask = None
                        if stats is not None:
                            stats.mask_wait_times.append(wait_s)
                            stats.mask_compute_times.append(result_holder[1])
                            stats.overlap_count += 1
                        bias = result_holder[0]
                        # Adjust for vocab size mismatch between hint and actual.
                        if bias.shape[0] > actual_vocab:
                            bias = bias[:actual_vocab]
                        elif bias.shape[0] < actual_vocab:
                            pad = torch.ones(
                                actual_vocab - bias.shape[0], dtype=torch.bool,
                                device=bias.device,
                            )
                            bias = torch.cat([bias, pad])
                    else:
                        # Fallback: synchronous (checker state changed after violation).
                        t_mask = time.perf_counter()
                        bias = checker.compute_mask(vocab_size=actual_vocab)
                        if stats is not None:
                            stats.mask_compute_times.append(time.perf_counter() - t_mask)
                    logits_with_noise[0, consume_idx, bias] = -np.inf
                    x0[0, consume_idx] = torch.argmax(logits_with_noise[0, consume_idx])

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break

                remaining = n_scheduled - tokens_placed_this_step
                batch_k = min(current_batch, remaining, n_available)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)
                if select_indices.shape[0] == 0:
                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                    return

                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vocab_idx = x0[0, pos].item()
                    if logits_with_noise[0, pos, vocab_idx] == -np.inf:
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                    return

                tokens_placed_this_step += len(positions)
                if stats is not None:
                    stats.tokens_unmasked += len(positions)
                    stats.batch_sizes.append(len(positions))

                # ── Grammar check ────────────────────────────────────────────
                t_gc = time.perf_counter()
                new_idx, violator = _extend_prefix(checker, x, consume_idx, mask_id)
                if stats is not None:
                    stats.grammar_check_times.append(time.perf_counter() - t_gc)

                if violator < 0:
                    consume_idx = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                else:
                    total_violations += 1
                    consume_idx = new_idx   # checker is now at the violator position

                    # Checker state is about to change — discard stale async mask.
                    if pending_mask is not None:
                        pending_mask[0].join()
                        pending_mask = None

                    if checker.is_accepting():
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    # ── DP violation fix ─────────────────────────────────────
                    # The checker state is right before the violator. Run DP on
                    # the entire contiguous non-mask segment from that point.
                    seg_end = consume_idx
                    while seg_end < x.shape[1] and x[0, seg_end].item() != mask_id:
                        seg_end += 1

                    total_dp_calls += 1
                    dp_succeeded = False

                    if seg_end > consume_idx:
                        fixes = dp_fix_prefix(
                            checker.matcher.deep_copy(),
                            x, consume_idx, log_probs, mask_id, top_k=top_k_dp,
                        )

                        if fixes is not None:
                            for fpos, ftok in fixes:
                                x[0, fpos] = ftok
                            total_fixes += len(fixes)
                            if trace and fixes:
                                print(f"  DP fixed {len(fixes)} pos: "
                                      f"{[(p - gen_start, ftok) for p, ftok in fixes]}")

                            seg_tokens = [x[0, p].item() for p in range(consume_idx, seg_end)]
                            c = checker.matcher.try_consume_tokens(seg_tokens)
                            consume_idx += c
                            dp_succeeded = True   # DP ran — don't fall through to raw fallback
                            if c < len(seg_tokens):
                                # Partial advance: remask the stall position and let model retry.
                                x[0, consume_idx] = mask_id
                                resamples.append((consume_idx, time.monotonic() - start_time))
                                tokens_placed_this_step -= 1
                                if stats is not None:
                                    stats.resample_count += 1
                                    stats.tokens_unmasked -= 1
                                if len(resamples) >= max_resamples:
                                    yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                                    return

                    if not dp_succeeded:
                        # DP found no valid path: fall back to remasking the violator
                        # and letting the model retry (same as the original method).
                        x[0, violator] = mask_id
                        resamples.append((violator, time.monotonic() - start_time))
                        tokens_placed_this_step -= 1
                        if stats is not None:
                            stats.resample_count += 1
                            stats.tokens_unmasked -= 1

                        if len(resamples) >= max_resamples:
                            yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx
                            return

                    current_batch = 1

                # ── Completion checks ────────────────────────────────────────
                if not complete and checker.is_accepting():
                    gen_ids = x[0, gen_start:].tolist()
                    first_mask = next(
                        (j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids)
                    )
                    if first_mask >= consume_idx - gen_start:
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

                if not complete:
                    gen_ids = x[0, gen_start:].tolist()
                    if eos_id in gen_ids or eot_id in gen_ids:
                        eos_pos = next(
                            (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
                        )
                        if eos_pos is not None and mask_id not in gen_ids[:eos_pos]:
                            for j in range(eos_pos, len(gen_ids)):
                                x[0, gen_start + j] = gen_ids[eos_pos]
                            complete = True

            yield x, resamples, False, total_violations, total_fixes, total_dp_calls, consume_idx

        # ── Early block-loop exit ─────────────────────────────────────────────
        # If complete=True was set during this block's step loop, all remaining
        # blocks would only do forward passes with n_scheduled=0 (the sequence
        # is fully filled). Break now to avoid those wasted GPU calls.
        if complete:
            break

    # Clean up any pending async mask thread.
    if pending_mask is not None:
        pending_mask[0].join()
        pending_mask = None

    gen_ids = x[0, gen_start:].tolist()
    is_complete = False
    if eos_id in gen_ids or eot_id in gen_ids:
        eos_pos = next(
            (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
        )
        is_complete = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

    yield x, resamples, is_complete, total_violations, total_fixes, total_dp_calls, consume_idx

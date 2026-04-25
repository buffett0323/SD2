"""Dgrammar: adaptive batch generation with token-level grammar checking.

Uses llguidance for incremental left-to-right token validation.

The checking strategy: maintain a consumed prefix index. After placing tokens,
try to extend the consumed prefix. If a token breaks the prefix, remask just
that token and try alternatives. Valid prefix tokens are always kept.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from dgrammar.checker import TokenChecker


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def extend_prefix(checker: TokenChecker, x, consume_idx: int, mask_id: int):
    """Try to consume contiguous non-mask tokens starting from consume_idx.

    Returns (new_consume_idx, violator_pos_or_neg1).
    - If all contiguous tokens are valid: (advanced_idx, -1)
    - If a token fails: (advanced_past_valid, violator_absolute_pos)
    The checker state is advanced past all valid tokens in both cases.
    """
    tokens_to_consume = []
    pos = consume_idx
    while pos < x.shape[1]:
        tid = x[0, pos].item()
        if tid == mask_id:
            break
        tokens_to_consume.append(tid)
        pos += 1

    if not tokens_to_consume:
        return consume_idx, -1

    count = checker.matcher.try_consume_tokens(tokens_to_consume)
    if count == len(tokens_to_consume):
        # All tokens valid
        return consume_idx + count, -1
    else:
        # Token at index 'count' is the violator.
        # The first 'count' tokens are valid and consumed in the checker state.
        violator_pos = consume_idx + count
        return consume_idx + count, violator_pos


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    checker: TokenChecker,
    prompt_len: int,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_id=126081,
    eot_id=126348,
    trace=False,
    max_batch_size=8,
    max_resamples=100,
):
    """Dgrammar with incremental token-level grammar checking."""
    start_time = time.monotonic()

    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
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

    # Consume prompt suffix tokens (between prompt_len and gen_start)
    if prompt_len < gen_start:
        prefix_tokens = x[0, prompt_len:gen_start].tolist()
        if not checker.consume_tokens(prefix_tokens):
            if trace:
                print("Warning: prompt suffix rejected by checker")

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
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

                mask_index = x == mask_id
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, block_end:] = -np.inf

                # Frontier masking: if consume_idx is a mask, apply grammar bias
                # so the token placed there is guaranteed grammar-valid.
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    bias = checker.compute_mask(vocab_size=logits_with_noise.shape[-1])
                    # bias is True where blocked. Set blocked tokens to -inf.
                    logits_with_noise[0, consume_idx, bias] = -np.inf
                    # Recompute x0 for this position
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
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                # Place tokens
                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vocab_idx = x0[0, pos].item()
                    if logits_with_noise[0, pos, vocab_idx] == -np.inf:
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                tokens_placed_this_step += len(positions)

                if trace:
                    placed_info = [(p - gen_start, x[0, p].item(), tokenizer.decode([x[0, p].item()])) for p in positions]
                    print(f"  Placed {len(positions)} at offsets: {placed_info}")

                # Grammar check: extend prefix, resolve violations
                total_grammar_checks += 1
                new_idx, violator = extend_prefix(checker, x, consume_idx, mask_id)

                if violator < 0:
                    # All valid
                    if new_idx > consume_idx and trace:
                        print(f"  Prefix advanced {consume_idx - gen_start} -> {new_idx - gen_start}")
                    consume_idx = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                else:
                    # Violation at specific position. Valid tokens before it are kept.
                    total_violations += 1
                    consume_idx = new_idx  # advance past valid tokens

                    if trace:
                        vt = x[0, violator].item()
                        print(f"  VIOLATION at offset {violator - gen_start}: "
                              f"token={vt} text={repr(tokenizer.decode([vt]))}")

                    # Check if grammar is already satisfied before retrying
                    if checker.is_accepting():
                        if trace:
                            print(f"  Grammar accepting at offset {consume_idx - gen_start}, completing early")
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    # Remask the violator and try alternatives
                    bad_token = x[0, violator].item()
                    logits_with_noise[0, violator, bad_token] = -np.inf
                    x[0, violator] = mask_id
                    total_remasks += 1
                    tokens_placed_this_step -= 1
                    resamples.append((violator, time.monotonic() - start_time))

                    if len(resamples) >= max_resamples:
                        yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                        return

                    # Try next-best tokens at the violator position
                    found = False
                    while len(resamples) < max_resamples:
                        next_vocab = torch.argmax(logits_with_noise[0, violator]).item()
                        if logits_with_noise[0, violator, next_vocab] == -np.inf:
                            break

                        # Try consuming just this one token
                        total_grammar_checks += 1
                        c = checker.matcher.try_consume_tokens([next_vocab])
                        if c == 1:
                            x[0, violator] = next_vocab
                            consume_idx += 1
                            tokens_placed_this_step += 1
                            found = True
                            if trace:
                                print(f"  Replaced with token={next_vocab} "
                                      f"text={repr(tokenizer.decode([next_vocab]))}")
                            # Try to extend further past the replacement
                            further_idx, further_viol = extend_prefix(
                                checker, x, consume_idx, mask_id
                            )
                            if further_viol < 0:
                                consume_idx = further_idx
                            else:
                                # Another violation further ahead, handle in next iteration
                                consume_idx = further_idx
                            break

                        logits_with_noise[0, violator, next_vocab] = -np.inf
                        total_remasks += 1
                        resamples.append((violator, time.monotonic() - start_time))

                    if not found:
                        # No valid alternative. Leave masked, model will reconsider.
                        if trace:
                            print(f"  No valid alternative for offset {violator - gen_start}, leaving masked")

                    current_batch = 1

                # Check for EOS/grammar completion
                if not complete and checker.is_accepting():
                    # Grammar is satisfied, check if all prefix tokens are placed
                    gen_ids = x[0, gen_start:].tolist()
                    first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids))
                    if first_mask >= consume_idx - gen_start:
                        # All tokens up to consume_idx are placed and validated
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        if trace:
                            print(f"  Grammar complete at offset {consume_idx - gen_start}")

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

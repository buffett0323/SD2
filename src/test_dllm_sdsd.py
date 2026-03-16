"""
DLLM Integration Test: SDSD with LLaDA-8B-Instruct or Dream-7B

Tests constrained decoding (DINGO/SDSD) on real diffusion LLM outputs.
Requires GPU (~20GB VRAM for Dream, ~16GB for LLaDA) and:
  pip install torch transformers>=4.46  # Dream
  pip install transformers==4.38.2       # LLaDA (different version)

Run:
  python test_dllm_sdsd.py --model dream     # Dream-7B (default)
  python test_dllm_sdsd.py --model llada     # LLaDA-8B-Instruct
  python test_dllm_sdsd.py --mock            # No GPU: use synthetic logits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from csr_dfa import build_csr_from_transition_dict
from sparse_dingo import sparse_dingo_dp


def get_device() -> tuple[torch.device, bool]:
    """Return (device, has_gpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def load_dream_model(device: torch.device):
    """Load Dream-v0-Instruct-7B."""
    from transformers import AutoModel, AutoTokenizer

    model_path = "Dream-org/Dream-v0-Instruct-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    return model, tokenizer


def load_llada_model(device: torch.device):
    """Load LLaDA-8B-Instruct."""
    from transformers import AutoModel, AutoTokenizer

    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    return model, tokenizer


def get_block_logits_dream(
    model, tokenizer, prompt: str, block_length: int, device: torch.device
) -> tuple[list[list[float]], int]:
    """
    Run Dream forward pass with masked block, return probability vectors.
    Dream: logits at position i predict token at i+1.
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if inputs.attention_mask is not None else None

    mask_id = getattr(tokenizer, "mask_token_id", None) or getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.pad_token_id or 0

    prompt_len = input_ids.shape[1]
    block_ids = torch.full((1, block_length), mask_id, dtype=torch.long, device=device)
    full_ids = torch.cat([input_ids, block_ids], dim=1)

    if attention_mask is not None:
        attn = torch.ones(1, full_ids.shape[1], device=device, dtype=torch.long)
        attn[0, :prompt_len] = attention_mask[0]
    else:
        attn = None

    with torch.no_grad():
        try:
            out = model(full_ids, attention_mask=attn)
        except TypeError:
            out = model(full_ids)
    logits = out.logits

    prob_vectors = []
    for j in range(block_length):
        pos = prompt_len + j - 1
        if pos < 0:
            pos = 0
        logit_j = logits[0, pos, :].float()
        probs = F.softmax(logit_j, dim=-1).cpu().tolist()
        prob_vectors.append(probs)

    return prob_vectors, mask_id


def get_block_logits_llada(
    model, tokenizer, prompt: str, block_length: int, device: torch.device
) -> tuple[list[list[float]], int]:
    """
    Run LLaDA forward pass with masked block.
    LLaDA uses different input format - prompt + mask tokens.
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)

    # LLaDA mask token - check config
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.pad_token_id or 0

    prompt_len = input_ids.shape[1]
    block_ids = torch.full(
        (1, block_length), mask_id, dtype=torch.long, device=device
    )
    full_ids = torch.cat([input_ids, block_ids], dim=1)

    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits

    prob_vectors = []
    for j in range(block_length):
        pos = prompt_len + j - 1
        if pos < 0:
            pos = 0
        logit_j = logits[0, pos, :].float()
        probs = F.softmax(logit_j, dim=-1).cpu().tolist()
        prob_vectors.append(probs)

    return prob_vectors, mask_id


def get_synthetic_logits(vocab_size: int, block_length: int, seed: int = 42) -> list[list[float]]:
    """Synthetic probability vectors for testing without GPU."""
    torch.manual_seed(seed)
    prob_vectors = []
    for _ in range(block_length):
        logits = torch.randn(vocab_size)
        probs = F.softmax(logits, dim=-1).tolist()
        prob_vectors.append(probs)
    return prob_vectors


def build_simple_json_dfa(vocab_size: int) -> tuple[object, int, set]:
    """
    Minimal JSON-like DFA: { "key" : "value" }
    Token IDs: 0={, 1=", 2=key, 3=:, 4=value, 5=}
    For mock/synthetic: vocab may not have these IDs; we use first 10 tokens as structure.
    """
    transitions = {
        (0, 0): 1, (1, 1): 2, (2, 2): 3, (3, 1): 4, (4, 3): 5,
        (5, 1): 6, (6, 4): 7, (7, 1): 8, (8, 5): 9,
    }
    csr = build_csr_from_transition_dict(transitions, num_states=10, vocab_size=vocab_size)
    return csr, 0, {9}


def build_permissive_dfa(vocab_size: int, valid_tokens: list[int] | None = None) -> tuple[object, int, set]:
    """DFA that accepts any sequence of tokens from valid_tokens (default: first 100)."""
    if valid_tokens is None:
        valid_tokens = list(range(min(100, vocab_size)))
    # 0 -> 1 on first token, then 1 -> 1 (self-loop) for remaining tokens
    transitions = {(0, t): 1 for t in valid_tokens} | {(1, t): 1 for t in valid_tokens}
    csr = build_csr_from_transition_dict(transitions, num_states=2, vocab_size=vocab_size)
    return csr, 0, {1}


def run_constrained_decode(
    prob_vectors: list[list[float]],
    csr, start_state: int, live_states: set[int],
) -> tuple[list[int], float]:
    """Run SDSD (B2) constrained decoding."""
    result = sparse_dingo_dp(csr, prob_vectors, start_state, live_states)
    return result.tokens, result.probability


def main():
    parser = argparse.ArgumentParser(description="DLLM + SDSD integration test")
    parser.add_argument("--model", choices=["dream", "llada"], default="dream")
    parser.add_argument("--mock", action="store_true", help="Use synthetic logits (no GPU)")
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--prompt", default="Generate a JSON object with key 'name' and value 'test'.")
    parser.add_argument("--constraint", choices=["permissive", "json"], default="permissive",
                        help="permissive=any token from top-100, json=strict { \"key\":\"value\" }")
    args = parser.parse_args()

    device, has_gpu = get_device()
    print(f"Device: {device}, GPU: {has_gpu}")

    model, tokenizer = None, None
    if args.mock or not has_gpu:
        if not args.mock and not has_gpu:
            print("No GPU detected. Using --mock mode with synthetic logits.")
        vocab_size = 32000  # Typical LLM vocab
        prob_vectors = get_synthetic_logits(vocab_size, args.block_length)
        print(f"Using synthetic logits: vocab={vocab_size}, block_len={args.block_length}")
    else:
        print(f"Loading {args.model}...")
        if args.model == "dream":
            model, tokenizer = load_dream_model(device)
            prob_vectors, _ = get_block_logits_dream(
                model, tokenizer, args.prompt, args.block_length, device
            )
        else:
            model, tokenizer = load_llada_model(device)
            prob_vectors, _ = get_block_logits_llada(
                model, tokenizer, args.prompt, args.block_length, device
            )
        vocab_size = len(prob_vectors[0])
        print(f"Got logits: vocab={vocab_size}, block_len={len(prob_vectors)}")

    # Build constraint DFA
    if args.constraint == "json":
        csr, start_state, live_states = build_simple_json_dfa(vocab_size)
    else:
        csr, start_state, live_states = build_permissive_dfa(vocab_size)

    # Run constrained decode
    tokens, prob = run_constrained_decode(prob_vectors, csr, start_state, live_states)
    print(f"\nSDSD constrained decode: {len(tokens)} tokens, P={prob:.6f}")
    print(f"Token IDs: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")

    if not args.mock and has_gpu and model is not None and tokenizer is not None:
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Decoded: {decoded[:200]}...")

    print("\n[DLLM + SDSD test complete]")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

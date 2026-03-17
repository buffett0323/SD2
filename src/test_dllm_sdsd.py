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
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # Dream uses rope_type="default" when rope_scaling is null, but ROPE_INIT_FUNCTIONS
    # in newer transformers only has linear/dynamic/yarn/longrope/llama3. Add "default".
    if "default" not in ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None, **kwargs):
            base = getattr(config, "rope_theta", 10000.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * kwargs.get("partial_rotary_factor", 1.0))
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                    / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

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
    from transformers.modeling_utils import PreTrainedModel

    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Compatibility for transformers 5.x only. With 4.38.2 (recommended for LLaDA), no patches needed.
    needs_patch = hasattr(PreTrainedModel, "mark_tied_weights_as_initialized")
    if needs_patch:
        _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized

        def _patched_mark(self, loading_info):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = getattr(self, "_tied_weights_keys", None) or {}
            if not getattr(self, "_tie_weights_compat_patched", False):
                _orig_tie = self.tie_weights

                def _compat_tie(missing_keys=None, recompute_mapping=True):
                    try:
                        _orig_tie(missing_keys=missing_keys, recompute_mapping=recompute_mapping)
                    except TypeError:
                        _orig_tie()

                self.tie_weights = _compat_tie
                self._tie_weights_compat_patched = True
            return _orig_mark(self, loading_info)

        PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark

    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        # LLaDAConfig may lack use_cache (cached config / transformers version mismatch)
        if not hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return model, tokenizer
    finally:
        if needs_patch:
            PreTrainedModel.mark_tied_weights_as_initialized = _orig_mark


def _to_messages(prompt: str | list) -> list:
    """Convert prompt to chat messages. Handles json-mode-eval list format."""
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        return prompt
    return [{"role": "user", "content": str(prompt)}]


def get_block_logits_dream(
    model, tokenizer, prompt: str | list, block_length: int, device: torch.device,
    prefix_tokens: list[int] | None = None,
) -> tuple[list[list[float]], int]:
    """
    Run Dream forward pass with masked block, return probability vectors.
    Dream: logits at position i predict token at i+1.
    prompt: str or list of messages (json-mode-eval format).
    prefix_tokens: already-accepted tokens for multi-round generation.
    """
    messages = _to_messages(prompt)
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if inputs.attention_mask is not None else None

    mask_id = getattr(tokenizer, "mask_token_id", None) or getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.pad_token_id or 0

    orig_prompt_len = input_ids.shape[1]
    if prefix_tokens:
        prefix = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        block_ids = torch.cat([prefix, torch.full((1, block_length), mask_id, dtype=torch.long, device=device)], dim=1)
        prompt_len = orig_prompt_len + len(prefix_tokens)
    else:
        block_ids = torch.full((1, block_length), mask_id, dtype=torch.long, device=device)
        prompt_len = orig_prompt_len
    full_ids = torch.cat([input_ids, block_ids], dim=1)

    if attention_mask is not None:
        attn = torch.ones(1, full_ids.shape[1], device=device, dtype=torch.bfloat16)
        attn[0, :orig_prompt_len] = attention_mask[0].to(torch.bfloat16)
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


def get_logits_for_position_dream(
    model, tokenizer, prompt: str | list, prefix_tokens: list[int], device: torch.device
) -> tuple[list[float], int]:
    """
    Step-by-step: 1 forward → logits for next position only. NFE = 1 per call.
    Used for sequential (Baseline) decoding where NFE = block_length.
    prompt: str or list of messages (json-mode-eval format).
    """
    messages = _to_messages(prompt)
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if inputs.attention_mask is not None else None

    mask_id = getattr(tokenizer, "mask_token_id", None) or getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.pad_token_id or 0

    if prefix_tokens:
        prefix = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        block_ids = torch.cat([prefix, torch.full((1, 1), mask_id, dtype=torch.long, device=device)], dim=1)
    else:
        block_ids = torch.full((1, 1), mask_id, dtype=torch.long, device=device)

    full_ids = torch.cat([input_ids, block_ids], dim=1)

    if attention_mask is not None:
        attn = torch.ones(1, full_ids.shape[1], device=device, dtype=torch.bfloat16)
        attn[0, : input_ids.shape[1]] = attention_mask[0].to(torch.bfloat16)
    else:
        attn = None

    with torch.no_grad():
        try:
            out = model(full_ids, attention_mask=attn)
        except TypeError:
            out = model(full_ids)

    pos = full_ids.shape[1] - 2
    if pos < 0:
        pos = 0
    logit = out.logits[0, pos, :].float()
    probs = F.softmax(logit, dim=-1).cpu().tolist()
    return probs, mask_id


def get_logits_for_position_llada(
    model, tokenizer, prompt: str | list, prefix_tokens: list[int], device: torch.device
) -> tuple[list[float], int]:
    """
    Step-by-step: 1 forward → logits for next position only. NFE = 1 per call.
    prompt: str or list of messages (json-mode-eval format).
    """
    messages = _to_messages(prompt)
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)

    mask_id = getattr(model.config, "mask_token_id", None) or tokenizer.pad_token_id or 0

    if prefix_tokens:
        prefix = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        block_ids = torch.cat([prefix, torch.full((1, 1), mask_id, dtype=torch.long, device=device)], dim=1)
    else:
        block_ids = torch.full((1, 1), mask_id, dtype=torch.long, device=device)

    full_ids = torch.cat([input_ids, block_ids], dim=1)

    with torch.no_grad():
        out = model(full_ids)

    pos = full_ids.shape[1] - 2
    if pos < 0:
        pos = 0
    logit = out.logits[0, pos, :].float()
    probs = F.softmax(logit, dim=-1).cpu().tolist()
    return probs, mask_id


def get_block_logits_llada(
    model, tokenizer, prompt: str | list, block_length: int, device: torch.device,
    prefix_tokens: list[int] | None = None,
) -> tuple[list[list[float]], int]:
    """
    Run LLaDA forward pass with masked block.
    prompt: str or list of messages (json-mode-eval format).
    prefix_tokens: already-accepted tokens for multi-round generation.
    """
    messages = _to_messages(prompt)
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)

    # LLaDA mask token - check config
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.pad_token_id or 0

    prompt_len = input_ids.shape[1]
    if prefix_tokens:
        prefix = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        block_ids = torch.cat([prefix, torch.full((1, block_length), mask_id, dtype=torch.long, device=device)], dim=1)
        prompt_len = prompt_len + len(prefix_tokens)
    else:
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


def get_verify_logits_dream(
    model, tokenizer, prompt: str | list, context_tokens: list[int], device: torch.device
) -> list[list[float]]:
    """
    Verification forward: run model on prompt + context_tokens (no mask).
    prompt: str or list of messages (json-mode-eval format).
    """
    messages = _to_messages(prompt)
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if inputs.attention_mask is not None else None

    if not context_tokens:
        return []

    context = torch.tensor([context_tokens], dtype=torch.long, device=device)
    full_ids = torch.cat([input_ids, context], dim=1)

    if attention_mask is not None:
        attn = torch.ones(1, full_ids.shape[1], device=device, dtype=torch.bfloat16)
        attn[0, : input_ids.shape[1]] = attention_mask[0].to(torch.bfloat16)
    else:
        attn = None

    with torch.no_grad():
        try:
            out = model(full_ids, attention_mask=attn)
        except TypeError:
            out = model(full_ids)

    prompt_len = input_ids.shape[1]
    prob_vectors = []
    for i in range(len(context_tokens)):
        pos = prompt_len + i - 1
        if pos < 0:
            pos = 0
        logit = out.logits[0, pos, :].float()
        probs = F.softmax(logit, dim=-1).cpu().tolist()
        prob_vectors.append(probs)
    return prob_vectors


def get_verify_logits_llada(
    model, tokenizer, prompt: str | list, context_tokens: list[int], device: torch.device
) -> list[list[float]]:
    """Verification forward for LLaDA. prompt: str or list of messages."""
    messages = _to_messages(prompt)
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)

    if not context_tokens:
        return []

    context = torch.tensor([context_tokens], dtype=torch.long, device=device)
    full_ids = torch.cat([input_ids, context], dim=1)

    with torch.no_grad():
        out = model(full_ids)

    prompt_len = input_ids.shape[1]
    prob_vectors = []
    for i in range(len(context_tokens)):
        pos = prompt_len + i - 1
        if pos < 0:
            pos = 0
        logit = out.logits[0, pos, :].float()
        probs = F.softmax(logit, dim=-1).cpu().tolist()
        prob_vectors.append(probs)
    return prob_vectors


def get_synthetic_logits(
    vocab_size: int, block_length: int, seed: int = 42,
    prefix_tokens: list[int] | None = None,
) -> list[list[float]]:
    """Synthetic probability vectors for testing without GPU."""
    torch.manual_seed(seed + (len(prefix_tokens) if prefix_tokens else 0))
    prob_vectors = []
    for _ in range(block_length):
        logits = torch.randn(vocab_size)
        probs = F.softmax(logits, dim=-1).tolist()
        prob_vectors.append(probs)
    return prob_vectors


def get_synthetic_verify_logits(
    vocab_size: int, context_tokens: list[int], seed: int = 42
) -> list[list[float]]:
    """Synthetic verification logits for mock mode."""
    torch.manual_seed(seed + len(context_tokens))
    prob_vectors = []
    for _ in range(len(context_tokens)):
        logits = torch.randn(vocab_size)
        probs = F.softmax(logits, dim=-1).tolist()
        prob_vectors.append(probs)
    return prob_vectors


def get_synthetic_logits_for_position(
    vocab_size: int, prefix_tokens: list[int], seed: int = 42
) -> list[float]:
    """One position for step-by-step mock. NFE=1 per call (simulated)."""
    torch.manual_seed(seed + len(prefix_tokens))
    logits = torch.randn(vocab_size)
    return F.softmax(logits, dim=-1).tolist()


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


def build_json_dfa_from_tokenizer(tokenizer) -> tuple[object, int, set]:
    """
    Build JSON DFA using real tokenizer IDs. Maps common JSON structure tokens
    to a permissive JSON-like DFA. Uses K << N for O(K) advantage.
    """
    vocab_size = tokenizer.vocab_size
    # Get token IDs for JSON structure chars (single-token encodings)
    def _tid(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    t_lbrace = _tid("{")
    t_rbrace = _tid("}")
    t_quote = _tid('"')
    t_colon = _tid(":")
    t_comma = _tid(",")
    t_lbracket = _tid("[")
    t_rbracket = _tid("]")

    # Build transitions: permissive JSON - any token from valid set
    valid_tokens = []
    for t in [t_lbrace, t_rbrace, t_quote, t_colon, t_comma, t_lbracket, t_rbracket]:
        if t is not None:
            valid_tokens.append(t)

    # Add common alphanumeric / string tokens (first 1k that decode to printable)
    for tid in range(min(1000, vocab_size)):
        if tid not in valid_tokens:
            try:
                s = tokenizer.decode([tid])
                if s and len(s) <= 2 and s.isprintable():
                    valid_tokens.append(tid)
            except Exception:
                pass
        if len(valid_tokens) >= 200:  # Cap K for O(K) benefit
            break

    if not valid_tokens:
        valid_tokens = list(range(min(100, vocab_size)))

    # Permissive: start -> accept any valid token -> live state
    transitions = {(0, t): 1 for t in valid_tokens} | {(1, t): 1 for t in valid_tokens}
    csr = build_csr_from_transition_dict(transitions, num_states=2, vocab_size=vocab_size)
    return csr, 0, {1}


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

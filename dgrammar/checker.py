"""Token-level grammar checker using llguidance.

Replaces the character-level DFA intersection approach (rustformlang) with
llguidance's incremental token-level parser. ~9000x faster per check.
"""

import torch
from llguidance import LLMatcher, LLTokenizer, LLParserLimits
from huggingface_hub import hf_hub_download


class TokenChecker:
    """Grammar checker operating directly on token IDs."""

    def __init__(self, grammar: str, model_name: str = "GSAI-ML/LLaDA-8B-Instruct"):
        tokenizer_path = hf_hub_download(
            repo_id=model_name, filename="tokenizer.json"
        )
        self.tokenizer = LLTokenizer(tokenizer_path)

        import json
        try:
            json.loads(grammar)
            grm = LLMatcher.grammar_from_json_schema(grammar)
        except (json.JSONDecodeError, TypeError):
            grm = LLMatcher.grammar_from_lark(grammar)

        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err, "Grammar is not valid"

        self._grammar = grm
        limits = LLParserLimits(max_items_in_row=20000, step_max_items=600000)
        self.matcher = LLMatcher(self.tokenizer, grm, log_level=1, limits=limits)
        self._limits = limits

    def validate_tokens(self, token_ids: list[int]) -> bool:
        """Check if a token sequence is valid (can be extended to a complete valid string)."""
        if not token_ids:
            return True
        count = self.matcher.validate_tokens(token_ids)
        return count == len(token_ids)

    def compute_mask(self, vocab_size: int = 126464) -> torch.Tensor:
        """Compute context-dependent valid token mask for the current state.

        Returns a bool tensor of shape [vocab_size] where True = blocked.
        """
        import numpy as np
        bias = self.matcher.compute_logit_bias()
        arr = np.frombuffer(bias, dtype=np.uint8)
        # 0 = blocked, >0 = allowed
        mask = torch.ones(vocab_size, dtype=torch.bool)
        blocked = torch.from_numpy((arr == 0).copy())
        mask[:len(arr)] = blocked
        return mask

    def consume_tokens(self, token_ids: list[int]) -> bool:
        """Advance parser state by consuming tokens. Returns False if invalid."""
        if not token_ids:
            return True
        count = self.matcher.try_consume_tokens(token_ids)
        if count != len(token_ids):
            self.matcher.rollback(count)
            return False
        return True

    def rollback(self, count: int) -> bool:
        """Roll back the last `count` consumed tokens."""
        if count <= 0:
            return True
        return self.matcher.rollback(count)

    def is_accepting(self) -> bool:
        """Check if current state is a valid end state."""
        return self.matcher.is_accepting()

    def is_stopped(self) -> bool:
        """Check if parser has reached a terminal state."""
        return self.matcher.is_stopped()

    def reset(self):
        """Reset parser to initial state."""
        self.matcher.reset()

    def clone(self):
        """Create a new checker with the same grammar (fresh state)."""
        checker = TokenChecker.__new__(TokenChecker)
        checker.tokenizer = self.tokenizer
        checker._grammar = self._grammar
        checker._limits = self._limits
        checker.matcher = LLMatcher(
            self.tokenizer, self._grammar, log_level=1, limits=self._limits
        )
        return checker

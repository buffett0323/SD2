"""Dgrammar: Grammar-constrained decoding for diffusion LLMs."""

__version__ = "0.1.0"

from dgrammar.generate import generate
from dgrammar.dp_generate import generate_dp, dp_fix_prefix

__all__ = ["generate", "generate_dp", "dp_fix_prefix"]

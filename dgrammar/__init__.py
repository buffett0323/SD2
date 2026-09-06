"""Dgrammar: Grammar-constrained decoding for diffusion LLMs.

Submodules are resolved lazily (PEP 562) because they have disjoint
dependencies:

    generate, dp_generate  ->  llguidance   (reactive repair; the frontier oracle)
    fa_generate            ->  torch only   (proactive alignment; the automaton
                                             comes from outlines-core and is
                                             passed in by the caller)

Importing the package eagerly would make ``fa_generate`` require llguidance,
which it never calls. That matters beyond tidiness: not depending on the
llguidance oracle is exactly what lets the FA path run exact inference over an
explicit automaton, so an import-time coupling would misrepresent the design --
and would force the llguidance wheel into images that have no use for it.
"""

import importlib

__version__ = "0.1.0"

_LAZY = {
    "generate": "dgrammar.generate",
    "generate_dp": "dgrammar.dp_generate",
    "dp_fix_prefix": "dgrammar.dp_generate",
    "generate_fa": "dgrammar.fa_generate",
    "decode": "dgrammar.fa_generate",
    "finalize": "dgrammar.fa_generate",
    "ByteDFA": "dgrammar.fa_generate",
    "TokenDFA": "dgrammar.fa_generate",
    "FAStats": "dgrammar.fa_generate",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value          # cache so later lookups skip this path
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

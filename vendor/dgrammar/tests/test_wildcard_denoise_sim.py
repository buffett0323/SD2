"""
Simulate a single diffusion-style denoising step that fills **multiple** MASK
positions at once, then checks viability with ``wildcard_earley_verify``.

This mirrors block / parallel unmasking: one step assigns terminals to several
MASK slots jointly (not necessarily one MASK per micro-step).

Run from ``dgrammar`` package root:
    python tests/test_wildcard_denoise_sim.py
    python tests/test_wildcard_denoise_sim.py --bruteforce
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dgrammar.wildcard_earley import Grammar, MASK, Rule, wildcard_earley_verify


# ── Grammars (same spirit as test_wildcard_stage1) ────────────────────────────


def dyck_grammar() -> Grammar:
    """S → ( S ) | ε"""
    return Grammar(
        start="S",
        rules=[
            Rule("S", ("(", "S", ")")),
            Rule("S", ()),
        ],
    )


def abc_grammar() -> Grammar:
    """S → a b c"""
    return Grammar(
        start="S",
        rules=[Rule("S", ("a", "b", "c"))],
    )


# ── Core simulation ───────────────────────────────────────────────────────────


def mask_positions(seq: list[str]) -> list[int]:
    return [i for i, t in enumerate(seq) if t == MASK]


def denoise_multi_masks(
    prefix: list[str],
    positions: list[int],
    terminals: list[str],
    grammar: Grammar,
) -> tuple[list[str], bool]:
    """
    One denoising step: replace MASK at ``positions[k]`` with ``terminals[k]``.

    Returns ``(new_prefix, viable)`` where *viable* is
    ``wildcard_earley_verify(new_prefix, grammar, n_masks_remaining)``.
    """
    if len(positions) != len(terminals):
        raise ValueError("positions and terminals must have the same length")
    new = list(prefix)
    for i, t in zip(positions, terminals):
        if new[i] != MASK:
            raise ValueError(f"position {i} is not MASK: {new[i]!r}")
        new[i] = t
    n_masks = sum(1 for x in new if x == MASK)
    ok = wildcard_earley_verify(new, grammar, n_masks)
    return new, ok


def verify_prefix(prefix: list[str], grammar: Grammar) -> bool:
    """Convenience: ``total_masks`` = number of MASK tokens in *prefix*."""
    n = sum(1 for t in prefix if t == MASK)
    return wildcard_earley_verify(prefix, grammar, n)


# ── Demos ─────────────────────────────────────────────────────────────────────


def demo_dyck_parallel_unmask() -> None:
    """
    Pattern: ``(`` MASK MASK ``)`` — one step assigns both inner MASKs (e.g. to
    ``(`` and ``)``) so the inner ``S`` becomes ``( S )`` with inner ``S → ε``.
    """
    g = dyck_grammar()
    prefix = ["(", MASK, MASK, ")"]
    print("=== Dyck: parallel unmask of 2 MASKs inside ( · · ) ===")
    print(f"  start: {prefix}  viable={verify_prefix(prefix, g)}")

    new, ok = denoise_multi_masks(prefix, [1, 2], ["(", ")"], g)
    print(f"  step:  positions [1,2] ← ['(', ')']")
    print(f"  after: {new}  viable={ok}  (expect True: (()))")

    new2, ok2 = denoise_multi_masks(prefix, [1, 2], [")", "("], g)
    print(f"  bad:   positions [1,2] ← [')', '(']")
    print(f"  after: {new2}  viable={ok2}  (expect False)")


def demo_abc_partial_then_full() -> None:
    """
    ``a`` MASK MASK — first step fills one MASK, second step fills the rest
    (multi-mask step only on the last line).
    """
    g = abc_grammar()
    print("\n=== ABC: partial step then one step for remaining MASKs ===")
    s1 = ["a", MASK, MASK]
    print(f"  start: {s1}  viable={verify_prefix(s1, g)}")

    s2, ok2 = denoise_multi_masks(s1, [1], ["b"], g)
    print(f"  after 1 MASK @1→'b': {s2}  viable={ok2}")

    s3, ok3 = denoise_multi_masks(s2, [2], ["c"], g)
    print(f"  after 1 MASK @2→'c': {s3}  viable={ok3}  (full concrete)")


def demo_abc_one_step_all_masks() -> None:
    """All three symbols are MASK: one joint assignment step ``b``, ``c`` with fixed ``a``?"""
    g = abc_grammar()
    print("\n=== ABC: [MASK,MASK,MASK] → one step assigns all three ===")
    s0 = [MASK, MASK, MASK]
    print(f"  start: {s0}  viable={verify_prefix(s0, g)}")
    new, ok = denoise_multi_masks(s0, [0, 1, 2], ["a", "b", "c"], g)
    print(f"  one step → ['a','b','c']: {new}  viable={ok}")


def bruteforce_joint_assignments(
    prefix: list[str],
    grammar: Grammar,
    vocab: list[str],
    *,
    max_trials: int = 50_000,
) -> None:
    """
    For small |MASK| and |vocab|, enumerate all joint assignments to MASK slots
    and count how many pass ``wildcard_earley_verify`` (feasible "denoisings").
    """
    idxs = mask_positions(prefix)
    if not idxs:
        print("no MASK in prefix")
        return
    n_mask = len(idxs)
    total = len(vocab) ** n_mask
    if total > max_trials:
        print(
            f"skip bruteforce: {total} combinations > max_trials={max_trials}"
        )
        return

    good = 0
    for choice in itertools.product(vocab, repeat=n_mask):
        new = list(prefix)
        for i, pos in enumerate(idxs):
            new[pos] = choice[i]
        n_rem = sum(1 for t in new if t == MASK)
        if wildcard_earley_verify(new, grammar, n_rem):
            good += 1

    print(
        f"\n=== Bruteforce joint assignments ===\n"
        f"  prefix={prefix}\n"
        f"  MASK count={n_mask}, vocab={vocab}\n"
        f"  feasible assignments: {good} / {total}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulate multi-MASK denoising steps + wildcard Earley check"
    )
    p.add_argument(
        "--bruteforce",
        action="store_true",
        help="enumerate small vocab^|MASK| joint assignments for a toy prefix",
    )
    args = p.parse_args()

    demo_dyck_parallel_unmask()
    demo_abc_partial_then_full()
    demo_abc_one_step_all_masks()

    if args.bruteforce:
        bruteforce_joint_assignments(
            ["(", MASK, MASK, ")"],
            dyck_grammar(),
            ["(", ")"],
        )
        bruteforce_joint_assignments(
            ["a", MASK, MASK],
            abc_grammar(),
            ["a", "b", "c"],
        )


if __name__ == "__main__":
    main()

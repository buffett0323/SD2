"""
Benchmark: optimized ``wildcard_earley_verify`` vs a **naive** reference
implementation (set[int] masks_used + full-chart fixpoint until convergence).

The naive version matches the semantics described in ``wildcard_earley.py``;
it is intentionally slower so you can measure the gap on harder inputs.

Run from ``dgrammar`` package root::

    python tests/bench_wildcard_earley_timing.py
    python tests/bench_wildcard_earley_timing.py --quick   # smaller inputs, faster naive
    python tests/bench_wildcard_earley_timing.py --optimized-only   # only optimized (huge n)

**Interpreting speedup:** ``speedup = naive_ms / optimized_ms`` (values ``>1`` mean the
optimized build is faster). Deep **Dyck** (concrete or many MASKs) and **multibracket**
typically show ~1.3–1.6×. **S → aⁿ** with all MASKs has a tiny chart—timings are noise-level
and naive can win. Very long **ambiguous expr** chains can also favor naive on some machines;
for a stable “optimized wins” story, prioritize the Dyck stress rows.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dgrammar.wildcard_earley import (
    MASK,
    Grammar,
    Rule,
    wildcard_earley_verify,
)

_StateKey = Tuple[Rule, int, int]
_NaiveChart = List[Dict[_StateKey, Set[int]]]


def _fixpoint_naive(
    pos: int,
    chart: _NaiveChart,
    grammar: Grammar,
    total_masks: int,
) -> None:
    """Prediction + completion until no change (full rescan each round)."""
    while True:
        changed = False
        for state in list(chart[pos].keys()):
            masks_set = chart[pos].get(state)
            if not masks_set:
                continue
            rule, dot, origin = state

            if dot < len(rule.rhs) and grammar.is_nonterminal(rule.rhs[dot]):
                next_sym = rule.rhs[dot]
                for new_rule in grammar.rules_for(next_sym):
                    new_state: _StateKey = (new_rule, 0, pos)
                    old = chart[pos].get(new_state, set())
                    merged = old | {0}
                    if merged != old:
                        chart[pos][new_state] = merged
                        changed = True

            if dot == len(rule.rhs):
                completed_sym = rule.lhs
                for (prev_rule, prev_dot, prev_orig), prev_masks in list(
                    chart[origin].items()
                ):
                    if (
                        prev_dot < len(prev_rule.rhs)
                        and prev_rule.rhs[prev_dot] == completed_sym
                    ):
                        combined: Set[int] = set()
                        for a in prev_masks:
                            for b in masks_set:
                                s = a + b
                                if s <= total_masks:
                                    combined.add(s)
                        if not combined:
                            continue
                        new_state = (prev_rule, prev_dot + 1, prev_orig)
                        old = chart[pos].get(new_state, set())
                        merged = old | combined
                        if merged != old:
                            chart[pos][new_state] = merged
                            changed = True

        if not changed:
            break


def _is_extendable_naive(
    chart_pos: Dict[_StateKey, Set[int]],
    start: str,
    total_masks: int,
) -> bool:
    for (rule, dot, origin), masks in chart_pos.items():
        if rule.lhs == start and origin == 0 and total_masks in masks:
            return True
    return False


def wildcard_earley_verify_naive(
    incomplete_prefix: List[str],
    grammar: Grammar,
    total_masks: int,
) -> bool:
    """Same semantics as ``wildcard_earley_verify``; intentionally slower reference."""
    n = len(incomplete_prefix)
    chart: _NaiveChart = [defaultdict(set) for _ in range(n + 1)]

    for rule in grammar.rules_for(grammar.start):
        chart[0][(rule, 0, 0)].add(0)

    _fixpoint_naive(0, chart, grammar, total_masks)

    for i, token in enumerate(incomplete_prefix):
        if token == MASK:
            for (rule, dot, origin), masks_set in chart[i].items():
                if dot < len(rule.rhs) and grammar.is_terminal(rule.rhs[dot]):
                    new_masks = {m + 1 for m in masks_set if m + 1 <= total_masks}
                    if new_masks:
                        chart[i + 1][(rule, dot + 1, origin)] |= new_masks
        else:
            for (rule, dot, origin), masks_set in chart[i].items():
                if dot < len(rule.rhs) and rule.rhs[dot] == token:
                    chart[i + 1][(rule, dot + 1, origin)] |= set(masks_set)

        _fixpoint_naive(i + 1, chart, grammar, total_masks)

    return _is_extendable_naive(chart[n], grammar.start, total_masks)


# ── Grammar builders (harder cases) ──────────────────────────────────────────


def dyck_grammar() -> Grammar:
    return Grammar(
        start="S",
        rules=[
            Rule("S", ("(", "S", ")")),
            Rule("S", ()),
        ],
    )


def multibracket_grammar() -> Grammar:
    return Grammar(
        start="S",
        rules=[
            Rule("S", ()),
            Rule("S", ("(", "S", ")")),
            Rule("S", ("[", "S", "]")),
        ],
    )


def fixed_a_chain_grammar(length: int) -> Grammar:
    return Grammar(
        start="S",
        rules=[Rule("S", tuple("a" * length))],
    )


def expr_grammar() -> Grammar:
    return Grammar(
        start="E",
        rules=[
            Rule("E", ("E", "+", "T")),
            Rule("E", ("T",)),
            Rule("T", ("id",)),
            Rule("T", ("(", "E", ")")),
        ],
    )


# ── Timing ────────────────────────────────────────────────────────────────────


def _bench_one(
    name: str,
    prefix: List[str],
    grammar: Grammar,
    total_masks: int,
    repeats: int,
    run_naive: bool,
) -> None:
    ok_opt = wildcard_earley_verify(prefix, grammar, total_masks)
    ok_naive = wildcard_earley_verify_naive(prefix, grammar, total_masks)
    if ok_opt != ok_naive:
        print(f"  MISMATCH {name}: opt={ok_opt} naive={ok_naive}")
        return

    # Warmup
    for _ in range(2):
        wildcard_earley_verify(prefix, grammar, total_masks)
        wildcard_earley_verify_naive(prefix, grammar, total_masks)

    t0 = time.perf_counter()
    for _ in range(repeats):
        wildcard_earley_verify(prefix, grammar, total_masks)
    t_opt = (time.perf_counter() - t0) / repeats

    t_naive_ms = None
    if run_naive:
        t1 = time.perf_counter()
        for _ in range(repeats):
            wildcard_earley_verify_naive(prefix, grammar, total_masks)
        t_naive = (time.perf_counter() - t1) / repeats
        t_naive_ms = t_naive * 1000
        ratio = t_naive / t_opt if t_opt > 0 else float("inf")
    else:
        ratio = None

    n_tok = len(prefix)
    print(f"  {name}")
    print(f"    n_tokens={n_tok}  n_masks={total_masks}  result={ok_opt}")
    print(f"    optimized: {t_opt * 1000:.4f} ms/call  ({repeats} runs)")
    if run_naive and t_naive_ms is not None and ratio is not None:
        print(f"    naive:     {t_naive_ms:.4f} ms/call")
        print(f"    speedup:   {ratio:.2f}x (naive / optimized)")


def build_cases(quick: bool) -> tuple[int, list[tuple[str, List[str], Grammar, int]]]:
    """Returns (repeats, cases)."""
    g_dyck = dyck_grammar()
    g_mb = multibracket_grammar()
    g_expr = expr_grammar()

    if quick:
        d_dyck = 80
        d_dyck_stress = 150
        d_mb = 60
        chain_n = 24
        mask_k = 16
        expr_reps = 40
        repeats = 15
    else:
        d_dyck = 220
        d_dyck_stress = 500
        d_mb = 160
        chain_n = 64
        mask_k = 40
        expr_reps = 200
        repeats = 25

    cases: list[tuple[str, List[str], Grammar, int]] = []

    # 1) Deep balanced Dyck, no MASK — large chart, many completions
    cases.append(
        (
            f"Dyck balanced depth {d_dyck} (concrete only)",
            ["("] * d_dyck + [")"] * d_dyck,
            g_dyck,
            0,
        )
    )

    cases.append(
        (
            f"Dyck balanced depth {d_dyck_stress} (concrete only, stress)",
            ["("] * d_dyck_stress + [")"] * d_dyck_stress,
            g_dyck,
            0,
        )
    )

    # 2) Many MASKs in a row, Dyck (short witness: nested pairs)
    cases.append(
        (
            f"Dyck {mask_k} consecutive MASKs",
            [MASK] * mask_k,
            g_dyck,
            mask_k,
        )
    )

    # 3) Fixed-a chain: S → a^n, all MASK — maximally ambiguous wildcard count
    g_chain = fixed_a_chain_grammar(chain_n)
    cases.append(
        (
            f"A-chain length {chain_n} (all MASK)",
            [MASK] * chain_n,
            g_chain,
            chain_n,
        )
    )

    # 4) Multi-bracket long valid nest + tail MASKs
    half = d_mb // 2
    mb_prefix = (
        ["(", "["] * half
        + ["]", ")"] * half
        + [MASK, MASK]
    )
    cases.append(
        (
            f"Multibracket interleaved depth ~{half} + 2 MASK",
            mb_prefix,
            g_mb,
            2,
        )
    )

    # 5) Expr: long left-associative chain of additions (ambiguous grammar)
    expr_tok: List[str] = []
    for _ in range(expr_reps):
        expr_tok.extend(["id", "+"])
    expr_tok.append("id")
    cases.append(
        (
            f"Expr left-chain {expr_reps} additions (concrete)",
            expr_tok,
            g_expr,
            0,
        )
    )

    return repeats, cases


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark wildcard Earley: optimized vs naive")
    p.add_argument(
        "--quick",
        action="store_true",
        help="smaller inputs so naive finishes in reasonable time",
    )
    p.add_argument(
        "--optimized-only",
        action="store_true",
        help="only time optimized (skip naive; for huge inputs)",
    )
    args = p.parse_args()

    repeats, cases = build_cases(quick=args.quick)
    run_naive = not args.optimized_only

    print("=== Wildcard Earley: optimized vs naive ===\n")
    if args.quick:
        print("(quick mode: smaller inputs)\n")
    if args.optimized_only:
        print("(naive skipped)\n")

    for name, prefix, grammar, total_masks in cases:
        _bench_one(name, prefix, grammar, total_masks, repeats, run_naive)
        print()

    print("Done.")


if __name__ == "__main__":
    main()

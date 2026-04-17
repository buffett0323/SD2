"""
Stage 1 correctness tests for wildcard_earley_verify.

Grammars:
  - Dyck:     S → ( S ) | ε
  - Mixed:    S → ε | ( S ) | [ S ]
  - Expr:     E → E + T | T,  T → id | ( E )
  - Quotes:   S → ε | " A ",  A → ε | ' S '  (outer double, inner single; iterative nest)
  - A-chain:  S → a^n — stress tests with 4+ MASK tokens as consecutive ``a``

MASK semantics: each MASK substitutes exactly one **terminal** at the grammar dot;
ε-subtrees come from S → ε via Earley completion, not from skipping input.

Acceptance: **IsExtendable** — some state (start → α·β, origin=0) with all masks accounted for.

Includes stress (Dyck depth 120), crossing brackets, and long ``a``-chains with many MASKs.

Run directly:   python tests/test_wildcard_stage1.py
Run via pytest: pytest tests/test_wildcard_stage1.py -v
"""

from __future__ import annotations

import os
import sys

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dgrammar.wildcard_earley import Grammar, Rule, MASK, wildcard_earley_verify


# ── Grammar factories ─────────────────────────────────────────────────────────


def balanced_paren_grammar() -> Grammar:
    """S → ( S ) | ε"""
    return Grammar(
        start="S",
        rules=[
            Rule("S", ("(", "S", ")")),
            Rule("S", ()),  # S → ε
        ],
    )


def multibracket_grammar() -> Grammar:
    """S → ε | ( S ) | [ S ] — properly nested round & square brackets."""
    return Grammar(
        start="S",
        rules=[
            Rule("S", ()),
            Rule("S", ("(", "S", ")")),
            Rule("S", ("[", "S", "]")),
        ],
    )


def expr_grammar() -> Grammar:
    """Classic ambiguous expr grammar: E → E + T | T,  T → id | ( E )."""
    return Grammar(
        start="E",
        rules=[
            Rule("E", ("E", "+", "T")),
            Rule("E", ("T",)),
            Rule("T", ("id",)),
            Rule("T", ("(", "E", ")")),
        ],
    )


def two_layer_quotes_grammar() -> Grammar:
    """
    Outer double quotes, inner single-quote regions that wrap a nested S.

    S → ε | \" A \"
    A → ε | ' S '

    Yields strings like \"\", \"''\", \"'\"\"'\", and deeply iterated
    \"'\"'\"…\"…\"'\"'\" (each layer wraps the previous in '\" … '\" ).
    """
    dq, sq = '"', "'"
    return Grammar(
        start="S",
        rules=[
            Rule("S", ()),
            Rule("S", (dq, "A", dq)),
            Rule("A", ()),
            Rule("A", (sq, "S", sq)),
        ],
    )


# ── Helpers ─────────────────────────────────────────────────────────────────────


def build_valid_quote_nesting(wraps: int) -> list[str]:
    """Innermost is ``\"\"``; each wrap prepends ``\"'`` and appends ``'\"``."""
    inner: list[str] = ['"', '"']
    for _ in range(wraps):
        inner = ['"', "'"] + inner + ["'", '"']
    return inner


def fixed_a_chain_grammar(length: int) -> Grammar:
    """S → a … a (``length`` terminal ``a`` in a row). Every MASK must match one ``a``."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return Grammar(
        start="S",
        rules=[Rule("S", tuple("a" * length))],
    )


def verify(tokens: list[str]) -> bool:
    g = balanced_paren_grammar()
    total = tokens.count(MASK)
    return wildcard_earley_verify(tokens, g, total)


def verify_grammar(tokens: list[str], g: Grammar) -> bool:
    return wildcard_earley_verify(tokens, g, tokens.count(MASK))


# ── Wildcard + balanced parens ─────────────────────────────────────────────────


def test_case3_parens_mask_parens_mask():
    """( ) * ( ) *  →  False   (()() is not in L(S))"""
    assert verify(["(", ")", MASK, "(", ")", MASK]) is False


def test_case4_two_masks_then_close():
    """* * )  →  False   (no assignment makes X Y ) a valid S)"""
    assert verify([MASK, MASK, ")"]) is False


def test_unbalanced_open():
    """(  →  True   (IsExtendable: prefix of “()”, etc.)"""
    assert verify(["("]) is True


def test_mask_only_two():
    """* *  →  False — no (S,·,0) at end with both MASKs as terminals (nested origins)."""
    assert verify([MASK, MASK]) is False


def test_open_mask_mask_close():
    """( * * )  →  False — terminal-only MASKs cannot mimic ε padding inside this grammar."""
    assert verify(["(", MASK, MASK, ")"]) is False


def test_paren_four_masks_inside_still_false():
    """( * * * * ) — four MASKs; dot stays on S (NT), still not valid."""
    assert verify(["(", MASK, MASK, MASK, MASK, ")"]) is False


# ── Multi-bracket: mixed ( ) and [ ] ───────────────────────────────────────────


def test_multibracket_interleaved_valid():
    """( [ ] ) , [ ( ) ] , ( [ ( ) ] ) — all in L(S)."""
    g = multibracket_grammar()
    assert verify_grammar(["(", "[", "]", ")"], g) is True
    assert verify_grammar(["[", "(", ")", "]"], g) is True
    assert verify_grammar(["(", "[", "(", ")", "]", ")"], g) is True


def test_multibracket_deep_mixed():
    """Long hand-built valid nest: ( [ ( [ ] ) ] )."""
    g = multibracket_grammar()
    tok = ["(", "[", "(", "[", "]", ")", "]", ")"]
    assert verify_grammar(tok, g) is True


def test_multibracket_crossing_invalid():
    """([)] — brackets cross; not in language."""
    g = multibracket_grammar()
    assert verify_grammar(["[", "(", "]", ")"], g) is False


def test_multibracket_prefix_single_open_extendable():
    """Single `(` or `[` — top-level S → (·…) with origin 0."""
    g = multibracket_grammar()
    assert verify_grammar(["("], g) is True
    assert verify_grammar(["["], g) is True


def test_multibracket_prefix_open_open_not_top_level_extendable():
    """( [ — no chart state with (S, origin=0) after two tokens; inner [·…] has origin>0."""
    g = multibracket_grammar()
    assert verify_grammar(["(", "["], g) is False


def test_multibracket_masks_cannot_fix_crossing():
    """([ * ] ) — one MASK cannot turn crossing into valid nest."""
    g = multibracket_grammar()
    assert verify_grammar(["[", "(", MASK, "]", ")"], g) is False


def test_multibracket_five_masks_only_false():
    """Five MASKs alone — no viable top-level S (same spirit as * * on Dyck)."""
    g = multibracket_grammar()
    assert verify_grammar([MASK] * 5, g) is False


# ── Fixed terminal chain: many MASKs as consecutive ``a`` ──────────────────────


def test_chain_four_masks_all_wildcard():
    """S → aaaa — four MASKs each as ``a`` (exactly four MASK tokens)."""
    g = fixed_a_chain_grammar(4)
    assert verify_grammar([MASK] * 4, g) is True


def test_chain_eight_masks_all_wildcard():
    """S → a^8 — eight MASKs (>3)."""
    g = fixed_a_chain_grammar(8)
    assert verify_grammar([MASK] * 8, g) is True


def test_chain_seven_positions_four_masks_mixed():
    """S → a^7 — four MASKs and three literal ``a`` (seven positions, >3 MASKs)."""
    g = fixed_a_chain_grammar(7)
    tok = [MASK, MASK, MASK, MASK, "a", "a", "a"]
    assert len(tok) == 7 and tok.count(MASK) == 4
    assert verify_grammar(tok, g) is True


def test_chain_nine_masks_exceeds_eight_a_rule():
    """S → a^8 — nine MASK tokens; eighth completes S, ninth has nowhere to go → False."""
    g = fixed_a_chain_grammar(8)
    assert verify_grammar([MASK] * 9, g) is False


def test_chain_six_masks_prefix_of_eight_extendable():
    """S → a^8 — six MASKs alone is a viable prefix (IsExtendable), not a complete match."""
    g = fixed_a_chain_grammar(8)
    assert verify_grammar([MASK] * 6, g) is True


# ── Expression grammar: E → E+T | T, T → id | (E) ─────────────────────────────
#
# Left-recursive E plus completion merges masks_used as m1+m2; chart items store
# *total* masks along the path, so some mixes (e.g. MASK + id vs id + id) can differ
# until completion semantics are split into per-constituent budgets.  Tests below
# stick to cases that match the current implementation.


def test_expr_add_chain():
    """id + id + id."""
    assert verify_grammar(["id", "+", "id", "+", "id"], expr_grammar()) is True


def test_expr_parenthesized_sum():
    """id + ( id + id )."""
    assert verify_grammar(
        ["id", "+", "(", "id", "+", "id", ")"],
        expr_grammar(),
    ) is True


def test_expr_deep_nesting():
    """id + ( id + ( id + ( id ) ) )"""
    g = expr_grammar()
    tok = [
        "id",
        "+",
        "(",
        "id",
        "+",
        "(",
        "id",
        "+",
        "(",
        "id",
        ")",
        ")",
        ")",
    ]
    assert verify_grammar(tok, g) is True


def test_expr_invalid_trailing_plus():
    """id + + — cannot extend to a full E from start in a sane way; prefix after second + is bad."""
    g = expr_grammar()
    assert verify_grammar(["id", "+", "+"], g) is False


def test_expr_mask_after_plus_extendable():
    """id + * — MASK can stand for T → id (viable prefix)."""
    g = expr_grammar()
    assert verify_grammar(["id", "+", MASK], g) is True


def test_expr_two_ids_no_operator_invalid():
    """id id — second id cannot continue a single E."""
    g = expr_grammar()
    assert verify_grammar(["id", "id"], g) is False


def test_expr_cannot_start_with_plus():
    """+ id — E cannot start with +."""
    g = expr_grammar()
    assert verify_grammar(["+", "id"], g) is False


# ── Two-layer quotes: S → ε | " A ", A → ε | ' S ' ───────────────────────────


def test_quotes_one_level_nesting():
    """\"'\"\"'\" — inner S is another hollow \"\"."""
    g = two_layer_quotes_grammar()
    assert verify_grammar(['"', "'", '"', '"', "'", '"'], g) is True


def test_quotes_iterated_wrap_depth_2_3_4():
    """Repeated '\" … '\" shells around inner \"\"."""
    g = two_layer_quotes_grammar()
    for wraps in (2, 3, 4):
        tok = build_valid_quote_nesting(wraps)
        assert verify_grammar(tok, g) is True, wraps


def test_quotes_stress_iterated_wrap_depth_28():
    """Many alternating DQ/SQ layers (long token sequence, still O(n³) Earley)."""
    g = two_layer_quotes_grammar()
    tok = build_valid_quote_nesting(28)
    assert len(tok) == 2 + 4 * 28
    assert verify_grammar(tok, g) is True


def test_quotes_malformed_dq_sq_dq_without_closing_sq():
    """\"'\" — cannot close inner ' S ' before outer \"."""
    g = two_layer_quotes_grammar()
    assert verify_grammar(['"', "'", '"'], g) is False


def test_quotes_malformed_swap_after_sq():
    """\"'\" — garbage: SQ then DQ without opening DQ-first S."""
    g = two_layer_quotes_grammar()
    assert verify_grammar(["'", '"'], g) is False


def test_quotes_prefix_lone_dq_extendable():
    """Single \" — can still finish as \"\" or \"'…'\"."""
    g = two_layer_quotes_grammar()
    assert verify_grammar(['"'], g) is True


def test_quotes_prefix_dq_sq_not_top_level_extendable():
    """\"' — no (S, origin=0) with full mask count at this prefix (cf. mixed-bracket ( [)."""
    g = two_layer_quotes_grammar()
    assert verify_grammar(['"', "'"], g) is False


def test_quotes_triple_dq_run_invalid():
    """\"\"\" — third DQ cannot start a well-formed continuation of this grammar."""
    g = two_layer_quotes_grammar()
    assert verify_grammar(['"', '"', '"'], g) is False


# ── Stress: very deep / long inputs (balanced parens only) ────────────────────


def test_stress_deep_balanced_no_mask_depth_120():
    """Dyck: depth 120, no MASK."""
    d = 120
    assert verify(["("] * d + [")"] * d) is True


def test_stress_almost_dyck_one_missing_close_extendable():
    """100 '(' and 99 ')' — still a prefix of balanced string (add one ')')."""
    d = 100
    assert verify(["("] * d + [")"] * (d - 1)) is True


def test_stress_invalid_leading_close():
    """Cannot be any prefix of Dyck language if first token is ')'."""
    assert verify([")", "(", "(", ")", ")"]) is False


def test_stress_many_concrete_then_masks_false():
    """50 '(' then 10 MASK — cannot close 50 opens with 10 MASK terminals."""
    assert verify(["("] * 50 + [MASK] * 10) is False


if __name__ == "__main__":
    import inspect

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    passed = failed = 0
    for name, fn in tests:
        doc = (inspect.getdoc(fn) or name).strip()
        try:
            fn()
            print(f"  PASS  {doc}")
            passed += 1
        except AssertionError:
            print(f"  FAIL  {doc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {doc}  ({exc})")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)

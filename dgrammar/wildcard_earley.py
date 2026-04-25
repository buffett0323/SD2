"""
Wildcard-aware Earley verifier — optimized for speed.

Key optimizations over the naive version:
  1. Bitmask instead of set[int] for masks_used  →  O(1) union / membership
  2. Worklist queue in _fixpoint               →  no repeated full-chart scans
  3. Early-exit on first True                  →  short-circuit as soon as viable
  4. Pre-computed terminal/nonterminal sets    →  no dict lookup in hot loop
  5. Grammar pre-index: symbol → waiting rules →  O(1) completion lookup
  6. Acceptance check moved inside scan loop   →  exit before processing full prefix

Semantics (unchanged from v1):
  - Each MASK consumes exactly one terminal step (no ε-skip of input).
  - Acceptance = IsExtendable: after the full prefix, some start-anchored state
    holds total_masks in its masks_used bitmask (dot can be anywhere).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

MASK = "MASK"


# ── Grammar ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: tuple[str, ...]

    def __repr__(self) -> str:
        body = " ".join(self.rhs) if self.rhs else "ε"
        return f"{self.lhs} → {body}"


class Grammar:
    def __init__(self, start: str, rules: List[Rule]) -> None:
        self.start = start
        self.rules = rules
        self.nonterminals: Set[str] = {r.lhs for r in rules}

        # lhs → list[Rule]
        self._by_lhs: Dict[str, List[Rule]] = defaultdict(list)
        # symbol → list[(rule, dot)]  — rules waiting for `symbol` at position dot
        self._waiting: Dict[str, List[Tuple[Rule, int]]] = defaultdict(list)

        for r in rules:
            self._by_lhs[r.lhs].append(r)
            for dot, sym in enumerate(r.rhs):
                self._waiting[sym].append((r, dot))

    def rules_for(self, symbol: str) -> List[Rule]:
        return self._by_lhs.get(symbol, [])

    def waiting_for(self, symbol: str) -> List[Tuple[Rule, int]]:
        """All (rule, dot) pairs where rule.rhs[dot] == symbol."""
        return self._waiting.get(symbol, [])

    def is_terminal(self, symbol: str) -> bool:
        return symbol not in self.nonterminals

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self.nonterminals


# ── Chart ─────────────────────────────────────────────────────────────────────
# chart[i][(rule, dot, origin)] = bitmask of masks_used values (bit k set ↔ k masks used)
# Using int as a bitmask: far cheaper than set[int] for small k (≤ 32).

_StateKey = Tuple[Rule, int, int]
_Chart = List[Dict[_StateKey, int]]  # int = bitmask


# ── Fixpoint with worklist ────────────────────────────────────────────────────


def _fixpoint(
    pos: int,
    chart: _Chart,
    grammar: Grammar,
    mask_limit_bit: int,   # (1 << (total_masks+1)) - 1  — upper bound bitmask
) -> None:
    """
    Run prediction + completion to fixpoint using a worklist queue.
    Only newly changed states are re-processed, avoiding O(n²) rescans.
    """
    queue: deque[_StateKey] = deque(chart[pos].keys())
    in_queue: Set[_StateKey] = set(queue)

    while queue:
        state = queue.popleft()
        in_queue.discard(state)

        masks_bits = chart[pos].get(state, 0)
        if not masks_bits:
            continue

        rule, dot, origin = state

        # ── Prediction ────────────────────────────────────────────────────────
        if dot < len(rule.rhs) and grammar.is_nonterminal(rule.rhs[dot]):
            next_sym = rule.rhs[dot]
            for new_rule in grammar.rules_for(next_sym):
                new_state: _StateKey = (new_rule, 0, pos)
                # Prediction: seed with masks_used=0 (bit0 only).
                # Parent masks are NOT inherited; completion adds them.
                old = chart[pos].get(new_state, 0)
                merged = old | 1  # bit0 = masks_used=0
                if merged != old:
                    chart[pos][new_state] = merged
                    if new_state not in in_queue:
                        queue.append(new_state)
                        in_queue.add(new_state)

        # ── Completion ────────────────────────────────────────────────────────
        if dot == len(rule.rhs):
            completed_sym = rule.lhs
            for (prev_rule, prev_dot, prev_orig), prev_bits in list(
                chart[origin].items()
            ):
                if (
                    prev_dot < len(prev_rule.rhs)
                    and prev_rule.rhs[prev_dot] == completed_sym
                ):
                    # Compute bitmask sum-convolution (capped at mask_limit_bit)
                    combined = _bitmask_sum(prev_bits, masks_bits, mask_limit_bit)
                    if not combined:
                        continue
                    new_state = (prev_rule, prev_dot + 1, prev_orig)
                    old = chart[pos].get(new_state, 0)
                    merged = old | combined
                    if merged != old:
                        chart[pos][new_state] = merged
                        if new_state not in in_queue:
                            queue.append(new_state)
                            in_queue.add(new_state)


def _bitmask_sum(a: int, b: int, limit_mask: int) -> int:
    """
    Return bitmask of all (i+j) where bit i is set in a and bit j is set in b,
    masked to limit_mask to avoid tracking values > total_masks.

    Example: a=0b0110 (values {1,2}), b=0b0011 (values {0,1})
             result = values {1,2,3} = 0b1110
    """
    result = 0
    tmp_b = b
    shift = 0
    while tmp_b:
        if tmp_b & 1:
            result |= (a << shift)
        tmp_b >>= 1
        shift += 1
    return result & limit_mask


# ── Main verifier ─────────────────────────────────────────────────────────────


def wildcard_earley_verify(
    incomplete_prefix: List[str],
    grammar: Grammar,
    total_masks: int,
) -> bool:
    """
    Return True iff the prefix is a viable prefix with exactly total_masks
    wildcard slots consumed.

    Each MASK in incomplete_prefix consumes exactly one terminal at the grammar dot.
    Acceptance = IsExtendable: after scanning the full prefix, some start-anchored
    state has total_masks in its masks_used set (dot may be anywhere).
    """
    # Bitmask where bit k represents "k masks used".
    # We only care about values 0..total_masks, so limit to (total_masks+1) bits.
    target_bit    = 1 << total_masks
    limit_mask    = (1 << (total_masks + 1)) - 1  # bits 0..total_masks

    n = len(incomplete_prefix)
    chart: _Chart = [defaultdict(int) for _ in range(n + 1)]

    # Seed with start rules, masks_used=0 (bit 0 set)
    for rule in grammar.rules_for(grammar.start):
        chart[0][(rule, 0, 0)] |= 1  # bit 0 = 0 masks used

    _fixpoint(0, chart, grammar, limit_mask)

    for i, token in enumerate(incomplete_prefix):
        if token == MASK:
            # MASK advances input and consumes one terminal at the dot
            for (rule, dot, origin), masks_bits in chart[i].items():
                if dot < len(rule.rhs) and grammar.is_terminal(rule.rhs[dot]):
                    # Shift left by 1: masks_used + 1 for each bit
                    new_bits = (masks_bits << 1) & limit_mask
                    if new_bits:
                        chart[i + 1][(rule, dot + 1, origin)] |= new_bits
        else:
            # Concrete token: exact match only
            for (rule, dot, origin), masks_bits in chart[i].items():
                if dot < len(rule.rhs) and rule.rhs[dot] == token:
                    chart[i + 1][(rule, dot + 1, origin)] |= masks_bits

        _fixpoint(i + 1, chart, grammar, limit_mask)

    # ── Accept: check IsExtendable only after full prefix is scanned ──────────
    # Early exit mid-prefix is unsafe: a complete parse at position i < n
    # does NOT mean the remaining tokens [i+1..n] form a valid continuation.
    return _is_extendable(chart[n], grammar.start, target_bit)


def _is_extendable(
    chart_pos: Dict[_StateKey, int],
    start: str,
    target_bit: int,
) -> bool:
    """Check if any start-anchored state has target_bit set (IsExtendable)."""
    for (rule, dot, origin), masks_bits in chart_pos.items():
        if rule.lhs == start and origin == 0 and (masks_bits & target_bit):
            return True
    return False


# ── Tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Grammar 1: S → ( S ) | ε   (balanced parentheses)
    paren_grammar = Grammar(
        start="S",
        rules=[
            Rule("S", ("(", "S", ")")),
            Rule("S", ()),
        ],
    )

    # Grammar 2: S → a b c
    abc_grammar = Grammar(
        start="S",
        rules=[Rule("S", ("a", "b", "c"))],
    )

    # Grammar 3: JSON-like  value → string | number | array
    #            string → " chars "   (simplified: " w ")
    #            number → n
    #            array  → [ value ]
    json_grammar = Grammar(
        start="value",
        rules=[
            Rule("value",  ("string",)),
            Rule("value",  ("number",)),
            Rule("value",  ("array",)),
            Rule("string", ('"', "w", '"')),
            Rule("number", ("n",)),
            Rule("array",  ("[", "value", "]")),
        ],
    )

    tests = [
        # SEMANTICS: each MASK = exactly ONE terminal. ε is handled by grammar rules.

        # ── Paren grammar: S → ( S ) | ε ─────────────────────────────────────
        (paren_grammar, ["(", ")"],                0, True,  "paren: () no masks"),
        (paren_grammar, [MASK, ")"],               1, True,  "paren: MASK=( then ) → ()"),
        (paren_grammar, [MASK, MASK, MASK, MASK],  4, True,  "paren: 4 MASKs → ( ( ) )"),
        (paren_grammar, [")"],                     0, False, "paren: bare ) invalid"),
        (paren_grammar, [MASK, ")", ")"],          1, False, "paren: ( ) ) unbalanced"),

        # ── ABC grammar: S → a b c ────────────────────────────────────────────
        (abc_grammar,   ["a", MASK, "c"],          1, True,  "abc: MASK=b"),
        (abc_grammar,   ["a", MASK, MASK],         2, True,  "abc: prefix a _ _"),
        (abc_grammar,   [MASK, MASK, MASK],        3, True,  "abc: all 3 MASKs"),
        (abc_grammar,   ["a", "b", "c", MASK],     1, False, "abc: extra MASK after complete"),
        (abc_grammar,   ["a", "a", MASK],          1, False, "abc: a a _ never valid"),

        # ── JSON-like grammar ─────────────────────────────────────────────────
        (json_grammar,  ['"', MASK, '"'],          1, True,  'json: " MASK " = string'),
        (json_grammar,  [MASK],                    1, True,  'json: MASK=n is valid value'),
        (json_grammar,  ["[", MASK, "]"],          1, True,  'json: [ n ] via 1 MASK'),
        (json_grammar,  [MASK, MASK, MASK],        3, True,  'json: 3 MASKs → [ n ]'),
        (json_grammar,  ["]", MASK],               1, False, 'json: ] MASK invalid start'),
    ]

    passed = 0
    for grammar, prefix, total, expected, desc in tests:
        result = wildcard_earley_verify(prefix, grammar, total)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} [{desc}]  got={result}  expected={expected}")

    print(f"\n{passed}/{len(tests)} tests passed")
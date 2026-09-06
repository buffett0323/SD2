"""Per-benchmark wiring for FA-constrained decoding.

Each task supplies a **regex**; the runner compiles it to a token-level
automaton with ``outlines_core.Index`` and hands that to
``dgrammar.fa_generate``.

Why regex and outlines rather than rustformlang
-----------------------------------------------
The constraint in ``vendor/constrained-diffusion`` is a two-level lexer+parser
(``CFG`` over lexeme names plus a ``BytesDFA`` per lexeme).  Its ``CFG`` exposes
no ``to_dfa``, and ``BytesDFA`` exposes no states or transitions -- only
predicates and ``to_text``.  There is therefore no path from that library to the
monolithic token automaton message passing needs.  ``outlines_core.Index`` is
built in Rust from a regex plus the real tokenizer vocabulary and exposes
``get_transitions() -> {state: {token: next_state}}`` directly, and it handles
the byte-level BPE correctly (tokens that are fragments of a multi-byte UTF-8
character), which a ``batch_decode`` round trip cannot.

Status
------
    JSON-Bench    ready.   Schema -> regex via outlines. S~190, E~390k median.
    SMILES-Bench  ready.   Depth-bounded regex below; depth 3 accepts 167/167
                           references, S=87, E=27k.
    CPP-Bench     blocked. Needs the real C++ grammar; see cpp_regex.

Measured, not assumed
---------------------
``bench/inspect_grammar.py`` reports max nesting depth 3 for SMILES and 5 for
C++ over the reference solutions, so bounded nesting is not a serious
restriction for either.  For SMILES the regex acceptance rate independently
agrees: depth 3 is exactly where coverage reaches 100%.

Validity is judged by an external checker, never by the automaton that produced
the string.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Regex sources
# ═══════════════════════════════════════════════════════════════════════════


def json_regex(instance, max_depth: int) -> str:
    schema = (getattr(instance, "data", {}) or {}).get("schema")
    if not schema:
        raise ValueError("instance has no schema")
    from outlines_core.json_schema import build_regex_from_schema

    return build_regex_from_schema(schema)


# Organic-subset atoms plus a *tight* bracket atom.  The tightness matters: a
# permissive `\[[^\]]{1,24}\]` pushes the lifted automaton past outlines' 2^31
# state limit, while this pattern keeps it at 87 states and still accepts every
# reference molecule.
_SMILES_BOND = r"[-=#:/\\.]"
_SMILES_RING = r"(?:%[0-9]{2}|[0-9])"
_SMILES_BRACKET = (
    r"\[(?:[0-9]{1,3})?(?:[A-Z][a-z]?|[bcnops])(?:@{1,2})?(?:H[0-9]?)?"
    r"(?:[+-][0-9]?)?\]"
)
_SMILES_ATOM = r"(?:Br|Cl|[BCNOPSFIbcnops]|%s)" % _SMILES_BRACKET


def _smiles_chain(depth: int) -> str:
    """SMILES chain regex with branch nesting bounded at ``depth``.

    Branch bodies must be full chains, not single atoms -- getting that wrong
    silently caps coverage at 77% instead of 100%, because it rejects any
    multi-atom branch such as ``(c1ccccc1)``.
    """
    if depth == 0:
        unit = r"(?:%s(?:%s)*)" % (_SMILES_ATOM, _SMILES_RING)
    else:
        branch = r"(?:\((?:%s)?%s\))" % (_SMILES_BOND, _smiles_chain(depth - 1))
        unit = r"(?:%s(?:%s)*(?:%s)*)" % (_SMILES_ATOM, _SMILES_RING, branch)
    return r"(?:(?:%s)?%s)+" % (_SMILES_BOND, unit)


def smiles_regex(instance, max_depth: int) -> str:
    return _smiles_chain(max_depth)


def cpp_regex(instance, max_depth: int) -> str:
    """NOT IMPLEMENTED -- and it should not be hand-written.

    ``inspect_grammar.py`` shows C++ reference bodies nest to at most depth 5,
    so a depth-bounded automaton is not the obstacle.  The obstacle is the
    grammar: C++ needs the real one, and the evidence against improvising is
    concrete -- the first hand-written SMILES regex looked reasonable and
    silently accepted only 77% of references, and a slightly permissive bracket
    atom blew past outlines' 2^31 state limit.  A hand-rolled C++ regex would
    fail the same way with far less chance of noticing.

    The grammar lives in ``constrained_diffusion.cfgs_our.cfg.get_cfg("cpp")``
    in the CD4dLLM vendor repo, which is not checked out here (only
    ``vendor/constrained-diffusion`` is).  With it, the route is: unroll only
    the self-embedding nonterminals to ``max_depth`` (left and right recursion
    express iteration and preserve regularity, so they stay), then emit a regex.

    Validate any implementation the same way SMILES was validated: compile it
    and check acceptance against every reference solution before spending GPU
    time.
    """
    raise NotImplementedError(
        "cpp_regex: needs the real C++ grammar from CD4dLLM "
        "(constrained_diffusion.cfgs_our.cfg.get_cfg('cpp')); see docstring."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Validity and functional checkers
# ═══════════════════════════════════════════════════════════════════════════


def json_valid(text: str, instance) -> bool:
    if not text or not text.strip():
        return False
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    schema = (getattr(instance, "data", {}) or {}).get("schema")
    if not schema:
        return True
    try:
        import jsonschema

        jsonschema.validate(obj, json.loads(schema))
        return True
    except Exception:  # noqa: BLE001
        return False


def json_functional(text: str, instance) -> Optional[bool]:
    ref = (getattr(instance, "data", {}) or {}).get("output")
    if ref is None or not text:
        return None
    try:
        return (json.dumps(json.loads(text), indent=4)
                == json.dumps(json.loads(ref), indent=4))
    except Exception:  # noqa: BLE001
        return False


def _rdkit_mol(smiles: str):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles)


def smiles_valid(text: str, instance) -> bool:
    """Chemical validity, not just grammatical.

    The automaton guarantees the string parses as SMILES; RDKit additionally
    checks it describes a real molecule (valences, aromaticity, ring closures
    that actually pair up). Without RDKit this degrades to a much weaker signal
    and the runner should say so rather than report a chemistry claim.
    """
    s = (text or "").strip()
    if not s:
        return False
    try:
        return _rdkit_mol(s) is not None
    except ImportError:
        return len(s) > 2


def smiles_functional(text: str, instance) -> Optional[bool]:
    ref = (getattr(instance, "data", {}) or {}).get("output")
    if ref is None or not text:
        return None
    try:
        from rdkit import Chem

        a, b = _rdkit_mol(text.strip()), _rdkit_mol(ref.strip())
        if a is None or b is None:
            return False
        return Chem.MolToSmiles(a) == Chem.MolToSmiles(b)
    except ImportError:
        return text.strip() == ref.strip()


def _run_gpp(source: str, args: list[str], timeout: float) -> tuple[bool, str]:
    if not shutil.which("g++"):
        return False, "g++ not available"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.cpp")
        with open(src, "w") as fh:
            fh.write(source)
        cmd = ["g++", "-std=c++17", src, *args]
        if "-fsyntax-only" not in args:
            cmd += ["-o", os.path.join(d, "m")]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        except subprocess.TimeoutExpired:
            return False, "compile timeout"
        if p.returncode != 0:
            return False, p.stderr[-400:]
        if "-fsyntax-only" in args:
            return True, ""
        try:
            r = subprocess.run([os.path.join(d, "m")], capture_output=True,
                               timeout=timeout, text=True)
        except subprocess.TimeoutExpired:
            return False, "run timeout"
        return r.returncode == 0, r.stderr[-400:]


def cpp_valid(text: str, instance) -> bool:
    """Syntactic validity -- ``extract_result`` already appends the tests."""
    if not text or not text.strip():
        return False
    ok, _ = _run_gpp(text, ["-fsyntax-only"], timeout=20.0)
    return ok


def cpp_functional(text: str, instance) -> Optional[bool]:
    if not text or not text.strip():
        return None
    ok, _ = _run_gpp(text, [], timeout=30.0)
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Task specs
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TaskSpec:
    """Everything the runner needs that varies across benchmarks."""

    name: str
    dataset: str
    lang: str
    regex: Callable[[object, int], str]
    valid: Callable[[str, object], bool]
    functional: Callable[[str, object], Optional[bool]]
    regular: bool
    fence_suffix: str = r"(?:\n```)?"
    per_instance_grammar: bool = False

    def build_regex(self, instance, max_depth: int) -> str:
        """Task regex plus an optional trailing code fence.

        The prompts ask for output inside ```` ```lang ... ``` ````, but whether
        the opening fence lands in the prompt or in the generation region depends
        on ``prepare_prompt``.  Making the closing fence optional is correct
        either way; guessing would silently constrain the model into a language
        it was not asked to produce.
        """
        return f"(?:{self.regex(instance, max_depth)}){self.fence_suffix}"


TASKS: dict[str, TaskSpec] = {
    "json": TaskSpec(
        name="JSON-Bench", dataset="jsonschema", lang="json",
        regex=json_regex, valid=json_valid, functional=json_functional,
        regular=True, per_instance_grammar=True,
    ),
    # The degeneracy venue.  JSON-Bench is easy enough that the phenomenon may
    # not appear there at all, whereas jsb_medium is where DPGrammar was
    # measured at 15.5% vacuous leaves against a ~2.5% baseline, and where the
    # existing DPGrammar/Dgrammar/LAVE runs live for comparison.  It carries no
    # reference outputs, so `functional` is absent by construction -- judge it
    # with the structural metrics in bench/measure_degeneracy.py instead.
    "jsb_medium": TaskSpec(
        name="JSONSchemaBench-medium", dataset="jsb_medium", lang="json",
        regex=json_regex, valid=json_valid, functional=lambda text, inst: None,
        regular=True, per_instance_grammar=True,
    ),
    "smiles": TaskSpec(
        name="SMILES-Bench", dataset="smiles", lang="smiles",
        regex=smiles_regex, valid=smiles_valid, functional=smiles_functional,
        regular=False,
    ),
    "cpp": TaskSpec(
        name="CPP-Bench", dataset="THUDM/humaneval-x/cpp", lang="cpp",
        regex=cpp_regex, valid=cpp_valid, functional=cpp_functional,
        regular=False, per_instance_grammar=True,
    ),
}

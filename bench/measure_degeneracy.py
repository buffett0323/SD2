"""Measure structural degeneracy in constrained-decoding outputs.

Separates the two failure populations identified in the DPGrammar analysis:

  Population A (model-inherent):  empty strings "", 1-char strings.
                                  Present in ALL methods at similar rates.
  Population B (repair-induced):  top-level {} / [], nested empty containers.
                                  Produced by the Viterbi repair path.

Usage
-----
    # single method
    python bench/measure_degeneracy.py --glob 'results/dp_jsb_medium_s0_t128*.jsonl'

    # compare methods on the instances all of them answered
    python bench/measure_degeneracy.py \
        --method DPGrammar 'results/dp_jsb_medium_s0_t128*.jsonl' \
        --method Dgrammar  'results/v2_async_ac4_timed*.jsonl' \
        --method LAVE      'results/lave_timed*.jsonl' \
        --common

    # non-JSON benchmarks
    python bench/measure_degeneracy.py --lang smiles --method FA 'results/fa_smiles*.jsonl'

Input format: JSONL with at least ``instance_id``, ``extracted``, ``valid``.
Optional fields used when present: ``time_taken``, ``resamples``, ``timing``.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import statistics
from typing import Any, Iterable


# ── loading ──────────────────────────────────────────────────────────────────


INFRA_REASONS = ("checker_init_error", "no_schema", "prompt_error")


def is_infra_failure(row: dict) -> bool:
    """Harness failure -- the schema never compiled, so nothing was decoded."""
    reason = row.get("timed_out_reason")
    return isinstance(reason, str) and reason.startswith(INFRA_REASONS)


def load(pattern: str) -> dict[str, dict]:
    """Load JSONL shards matching ``pattern``, keyed by instance_id (last wins)."""
    rows: dict[str, dict] = {}
    for path in sorted(globmod.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = r.get("instance_id")
                if iid is not None:
                    rows[iid] = r
    return rows


def text_of(row: dict) -> str | None:
    """The extracted output as a string, or None if there is none.

    Deliberately does NOT collapse internal whitespace.  It used to, which
    rewrote the data being measured -- "John Doe" became "JohnDoe" -- and
    understated median character counts across the board.  The regex fallbacks
    that relied on the collapsed form now tolerate whitespace themselves.
    """
    ex = row.get("extracted")
    if ex is None:
        return None
    s = ex if isinstance(ex, str) else json.dumps(ex)
    if not s.strip():
        return None
    return s


def n_resamples(row: dict) -> int:
    v = row.get("resamples")
    if isinstance(v, list):
        return len(v)
    return int(v or 0)


# ── JSON structural walk ─────────────────────────────────────────────────────


def walk_json(node: Any, acc: dict) -> None:
    """Accumulate leaf/container statistics over a parsed JSON tree."""
    if isinstance(node, dict):
        acc["containers"] += 1
        if not node:
            acc["empty_containers"] += 1
            acc["empty_objects"] += 1
        for v in node.values():
            walk_json(v, acc)
    elif isinstance(node, list):
        acc["containers"] += 1
        if not node:
            acc["empty_containers"] += 1
            acc["empty_arrays"] += 1
        for v in node:
            walk_json(v, acc)
    else:
        acc["leaves"] += 1
        if isinstance(node, str):
            acc["str_leaves"] += 1
            if len(node) == 0:
                acc["empty_strs"] += 1
            elif len(node) == 1:
                acc["onechar_strs"] += 1


def json_stats(s: str) -> dict | None:
    """Structural stats for one output, or None if it does not parse as JSON."""
    try:
        tree = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    acc = dict(
        containers=0, empty_containers=0, empty_objects=0, empty_arrays=0,
        leaves=0, str_leaves=0, empty_strs=0, onechar_strs=0,
    )
    walk_json(tree, acc)
    return acc


# ── per-language degeneracy predicates ───────────────────────────────────────

_ONECHAR_RE = re.compile(r':\s*"\w"\s*[,}\]]')


def score_json(s: str) -> dict:
    """Degeneracy flags for one JSON output. Regex fallbacks when parsing fails."""
    st = json_stats(s)
    out = {
        "top_empty": s.strip() in ("{}", "[]"),
        "parsed": st is not None,
    }
    if st is not None:
        out.update(
            has_empty_obj=st["empty_objects"] > 0,
            has_empty_arr=st["empty_arrays"] > 0,
            has_empty_str=st["empty_strs"] > 0,
            has_onechar=st["onechar_strs"] > 0,
            # fraction of leaves carrying no information
            vacuous_leaf_frac=(
                (st["empty_strs"] + st["onechar_strs"]) / st["leaves"]
                if st["leaves"] else 0.0
            ),
            empty_container_frac=(
                st["empty_containers"] / st["containers"] if st["containers"] else 0.0
            ),
            leaves=st["leaves"],
        )
    else:
        out.update(
            has_empty_obj="{}" in s,
            has_empty_arr="[]" in s,
            has_empty_str='""' in s,
            has_onechar=bool(_ONECHAR_RE.search(s)),
            vacuous_leaf_frac=0.0,
            empty_container_frac=0.0,
            leaves=0,
        )
    return out


def score_flat(s: str, trivial_len: int) -> dict:
    """Degeneracy flags for non-JSON languages (SMILES, C++).

    ``trivial_len`` is the length below which an output is considered content-free
    (e.g. a bare ``C`` for SMILES, an empty function body for C++).
    """
    return {
        "top_empty": len(s) <= trivial_len,
        "parsed": True,
        "has_empty_obj": "{}" in s,
        "has_empty_arr": "[]" in s,
        "has_empty_str": '""' in s,
        "has_onechar": False,
        "vacuous_leaf_frac": 0.0,
        "empty_container_frac": 0.0,
        "leaves": 0,
    }


# ── reporting ────────────────────────────────────────────────────────────────

COLS = [
    # NOT validity: the runner's `valid` field is its is_complete flag (EOS
    # placed, no MASK before it).  An unconstrained run finishes cleanly on
    # malformed JSON and scores here.  Use measure_valid.py for real validity.
    ("valid",       "completes",  lambda r, f: bool(r.get("valid"))),
    ("topEmpty",    "top-empty",  lambda r, f: f["top_empty"]),
    ("emptyObj",    "{} in",      lambda r, f: f["has_empty_obj"]),
    ("emptyArr",    "[] in",      lambda r, f: f["has_empty_arr"]),
    ("emptyStr",    '"" in',      lambda r, f: f["has_empty_str"]),
    ("oneChar",     "1char",      lambda r, f: f["has_onechar"]),
]


def summarize(name: str, rows: dict[str, dict], keys: Iterable[str], lang: str,
              trivial_len: int) -> dict:
    keys = list(keys)
    n_total = len(keys)
    # An output that does not exist has no empty brackets, so scoring it as
    # "not degenerate" would reward failing outright.  Score the outputs that
    # exist and report how many did not, rather than hiding the difference.
    keys = [k for k in keys if text_of(rows.get(k, {}))]
    n = len(keys)
    if n == 0:
        return {"name": name, "n": 0}
    n_missing = n_total - n

    counts = {c[0]: 0 for c in COLS}
    lengths, vacuous, empty_frac, times = [], [], [], []
    res_pos, res_zero = [], []

    for k in keys:
        r = rows[k]
        s = text_of(r)
        f = score_json(s) if lang == "json" else score_flat(s, trivial_len)
        for key, _label, fn in COLS:
            if fn(r, f):
                counts[key] += 1
        lengths.append(len(s))
        vacuous.append(f["vacuous_leaf_frac"])
        empty_frac.append(f["empty_container_frac"])
        if r.get("time_taken") is not None:
            times.append(float(r["time_taken"]))
        (res_pos if n_resamples(r) >= 1 else res_zero).append(f)

    def rate(key):
        return 100.0 * counts[key] / n

    out = {
        "name": name,
        "n": n,
        "n_total": n_total,
        "no_output_pct": 100.0 * n_missing / n_total if n_total else 0.0,
        **{key: rate(key) for key, _l, _f in COLS},
        "median_chars": statistics.median(lengths),
        "vacuous_leaf_pct": 100.0 * statistics.mean(vacuous),
        "empty_container_pct": 100.0 * statistics.mean(empty_frac),
        "median_time": statistics.median(times) if times else None,
        "mean_time": statistics.mean(times) if times else None,
    }

    # Dose-response: does degeneracy track repair activity?
    def frac_degen(group):
        if not group:
            return None
        hit = sum(1 for f in group if f["has_empty_obj"] or f["has_empty_arr"]
                  or f["has_empty_str"])
        return 100.0 * hit / len(group)

    out["degen_resample0"] = frac_degen(res_zero)
    out["degen_resample1plus"] = frac_degen(res_pos)
    out["n_resample0"] = len(res_zero)
    out["n_resample1plus"] = len(res_pos)
    return out


def print_table(results: list[dict], lang: str) -> None:
    results = [r for r in results if r["n"]]
    if not results:
        print("no rows")
        return

    hdr = f"{'method':<12}{'n':>5}" + "".join(f"{lbl:>10}" for _k, lbl, _f in COLS)
    hdr += f"{'vacLeaf':>9}{'medChar':>9}{'medTime':>9}{'noOut':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        line = f"{r['name']:<12}{r['n']:>5}"
        for key, _lbl, _fn in COLS:
            line += f"{r[key]:>9.1f}%"
        line += f"{r['vacuous_leaf_pct']:>8.1f}%"
        line += f"{r['median_chars']:>9.0f}"
        line += f"{r['median_time']:>9.1f}" if r["median_time"] is not None else f"{'-':>9}"
        line += f"{r.get('no_output_pct', 0.0):>7.1f}%"
        print(line)

    print()
    print("dose-response (contains any empty obj/arr/str, split by repair activity)")
    print(f"{'method':<12}{'resample==0':>22}{'resample>=1':>22}")
    print("-" * 56)
    for r in results:
        a, b = r.get("degen_resample0"), r.get("degen_resample1plus")
        sa = f"{a:.1f}% (n={r['n_resample0']})" if a is not None else "-"
        sb = f"{b:.1f}% (n={r['n_resample1plus']})" if b is not None else "-"
        print(f"{r['name']:<12}{sa:>22}{sb:>22}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", nargs=2, action="append", metavar=("NAME", "GLOB"),
                    help="named result set; repeatable")
    ap.add_argument("--glob", help="shorthand for a single unnamed method")
    ap.add_argument("--common", action="store_true",
                    help="restrict to instances every method produced output for")
    ap.add_argument("--keep-infra", action="store_true",
                    help="keep instances whose schema llguidance cannot compile")
    ap.add_argument("--lang", default="json", choices=["json", "smiles", "cpp"])
    ap.add_argument("--trivial-len", type=int, default=2,
                    help="non-JSON: outputs at or below this length count as empty")
    args = ap.parse_args()

    specs = args.method or []
    if args.glob:
        specs = specs + [("output", args.glob)]
    if not specs:
        ap.error("give --glob or at least one --method NAME GLOB")

    loaded = [(name, load(pattern)) for name, pattern in specs]
    for name, rows in loaded:
        if not rows:
            print(f"warning: no rows matched for {name}")

    # Harness failures -- llguidance could not compile the schema, so no method
    # ever ran.  Dropped from every arm at once and announced, so the
    # denominators stay identical.
    infra = sorted({k for _n, rows in loaded for k, r in rows.items()
                    if is_infra_failure(r)})
    if infra and not args.keep_infra:
        loaded = [(n, {k: v for k, v in rows.items() if k not in set(infra)})
                  for n, rows in loaded]
        print(f"excluding {len(infra)} instances whose schema llguidance cannot "
              f"compile (no method runs on them)")

    if args.common and len(loaded) > 1:
        # Intersect on instance_id, NOT on "produced text".  Filtering on text
        # here silently deleted each arm's own failures from the comparison,
        # and the arms fail at different rates -- so the arm that failed more
        # had more of its failures removed.
        keys = set.intersection(*[set(rows) for _n, rows in loaded])
        print(f"restricted to {len(keys)} instances answered by all "
              f"{len(loaded)} methods\n")
    else:
        keys = None

    results = [
        summarize(name, rows, keys if keys is not None else rows.keys(),
                  args.lang, args.trivial_len)
        for name, rows in loaded
    ]
    print_table(results, args.lang)


if __name__ == "__main__":
    main()

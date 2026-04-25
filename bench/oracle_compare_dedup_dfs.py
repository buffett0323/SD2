#!/usr/bin/env python3
"""
Compare ``oracle_verify_fast`` results: ``dfs`` vs ``bfs_dedup`` vs ``smart``.

Use this to estimate **disagreement rate** for dedup fingerprints (and whether
``smart`` matches ``dfs`` when phase-1 returns False).

Input: JSONL with one object per line::

    {"grammar": "<json schema string or lark>", "prefix": [int, ...], "block": [int, ...]}

``prefix`` + ``block`` are tokenizer ids (same convention as LAVE / ``oracle_fast``).
MASK positions use ``mask_id`` 126336 unless you override ``--mask-id``.

Example::

    cd bench && python oracle_compare_dedup_dfs.py --input cases.jsonl --limit 500

Env ``DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS`` is honored (refines dedup key mid-run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from oracle_fast import OracleState, oracle_verify_fast  # noqa: E402


def _run_mode(
    grammar: str,
    prefix: list[int],
    block: list[int],
    oracle_mask_limit: int,
    mode: str,
) -> tuple[bool | None, dict]:
    st = OracleState(grammar)
    timing: dict = {}
    r = oracle_verify_fast(
        st,
        prefix,
        block,
        oracle_mask_limit,
        search_mode=mode,  # type: ignore[arg-type]
        timing_out=timing,
        max_search_seconds=None,
    )
    return r, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Max lines (0 = all)")
    ap.add_argument("--oracle-mask-limit", type=int, default=32)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"Missing file: {path}", file=sys.stderr)
        sys.exit(1)

    n = 0
    skipped = 0
    dedup_vs_dfs = 0
    smart_vs_dfs = 0
    timeouts = 0

    with path.open() as f:
        for line in f:
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            grammar = row.get("grammar")
            prefix = row.get("prefix") or []
            block = row.get("block") or []
            if grammar is None or not isinstance(block, list):
                skipped += 1
                continue
            if not isinstance(prefix, list):
                skipped += 1
                continue

            rd, _ = _run_mode(
                grammar, prefix, block, args.oracle_mask_limit, "dfs"
            )
            rdd, _ = _run_mode(
                grammar, prefix, block, args.oracle_mask_limit, "bfs_dedup"
            )
            rs, _ = _run_mode(
                grammar, prefix, block, args.oracle_mask_limit, "smart"
            )

            if rd is None or rdd is None or rs is None:
                timeouts += 1
            elif rd != rdd:
                dedup_vs_dfs += 1
            if rd is not None and rs is not None and rd != rs:
                smart_vs_dfs += 1

            n += 1

    print(
        json.dumps(
            {
                "lines_ok": n,
                "skipped_malformed": skipped,
                "timeouts_any_mode": timeouts,
                "dedup_vs_dfs_disagreements": dedup_vs_dfs,
                "smart_vs_dfs_disagreements": smart_vs_dfs,
                "dedup_disagreement_rate": (dedup_vs_dfs / n) if n else 0.0,
                "smart_disagreement_rate": (smart_vs_dfs / n) if n else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

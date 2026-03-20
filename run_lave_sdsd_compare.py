#!/usr/bin/env python3
"""
Fair comparison: our SDSD / BiDi methods vs LAVE on the same setup.

Aligned knobs (defaults match ``run_unified_benchmark.py`` and Dgrammar/LAVE scripts):

- **Model**: LLaDA-8B-Instruct (loaded inside ``run_unified_benchmark.py``)
- **Dataset**: ``eth-sri/json-mode-eval-extended`` (jsonschema / JSON-Bench), via Hugging Face
- **Diffusion**: ``steps=128``, ``block_length=32``, ``gen_length=256``, ``temperature=0.2`` (temperature fixed in ``diffusion_sdsd.generate_diffusion_sdsd``)

LAVE is executed **only** if ``vendor/dgrammar/bench/run_lave_timed.py`` exists (same convention as ``run_unified_benchmark.sh``).

Examples::

    # Ours only (writes JSONL under results/lave_sdsd_compare)
    python run_lave_sdsd_compare.py --methods sdsd,bidi --limit 20

    # Ours + LAVE (needs vendor/dgrammar checkout) + aggregate table
    python run_lave_sdsd_compare.py --methods sdsd,bidi --limit 272 --run-lave --aggregate

    # Table from existing dirs
    python run_lave_sdsd_compare.py --aggregate-only \\
        --our-results results/lave_sdsd_compare \\
        --lave-results vendor/dgrammar/results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_STEPS = 128
DEFAULT_BLOCK = 32
DEFAULT_GEN = 256


def _print_shared_config(
    *,
    steps: int,
    block_length: int,
    gen_length: int,
    dataset: str,
    lave_seed: int,
) -> None:
    cfg = {
        "dataset_hf": dataset,
        "diffusion_steps": steps,
        "block_length": block_length,
        "gen_length": gen_length,
        "temperature_generate_diffusion_sdsd": 0.2,
        "lave_bench_seed_arg": lave_seed,
        "note": "LAVE script args: seed num_instances jsonschema T extra (see vendor/dgrammar/bench/run_lave_timed.py)",
    }
    print("\n=== Shared experiment config (SDSD vs LAVE) ===")
    print(json.dumps(cfg, indent=2))
    print()


def _run_unified_benchmark_subprocess(args: argparse.Namespace, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "run_unified_benchmark.py"),
        "--methods",
        args.methods,
        "--output",
        str(out_dir),
        "--steps",
        str(args.steps),
        "--block-length",
        str(args.block_length),
        "--gen-length",
        str(args.gen_length),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.skip_slow:
        cmd.append("--skip-slow")
    if args.mock:
        cmd.append("--mock")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_lave_vendor(args: argparse.Namespace) -> int:
    script = ROOT / "vendor" / "dgrammar" / "bench" / "run_lave_timed.py"
    if not script.is_file():
        print(
            f"\n[SKIP] LAVE: missing {script}\n"
            "  Clone constrained decoding stack per README (vendor/dgrammar + deps), then re-run with --run-lave.\n"
        )
        return 1
    n_inst = args.limit if args.limit is not None else 272
    cmd = [
        sys.executable,
        str(script),
        str(args.lave_seed),
        str(n_inst),
        "jsonschema",
        str(args.steps),
        str(args.lave_extra),
    ]
    dgrammar_root = script.parent.parent
    print("Running LAVE:", " ".join(cmd), f"(cwd={dgrammar_root})")
    return subprocess.call(cmd, cwd=str(dgrammar_root))


def _run_aggregate(our: Path, lave_dir: Path | None) -> int:
    agg = ROOT / "aggregate_unified_results.py"
    if not agg.is_file():
        print("Missing aggregate_unified_results.py")
        return 1
    cmd = [sys.executable, str(agg), str(our)]
    if lave_dir is not None and lave_dir.is_dir():
        cmd.append(str(lave_dir))
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description="SDSD/BiDi vs LAVE: same LLaDA + JSON-Bench + diffusion steps (see --help)."
    )
    p.add_argument(
        "--methods",
        default="sdsd,bidi",
        help="Comma-separated methods passed to run_unified_benchmark.py (default: sdsd,bidi)",
    )
    p.add_argument("--limit", type=int, default=None, help="Number of jsonschema instances (default: all ~272)")
    p.add_argument(
        "--output",
        default="results/lave_sdsd_compare",
        help="Directory for our JSONL (run_unified_benchmark --output)",
    )
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps / LAVE T (default 128)")
    p.add_argument("--block-length", type=int, default=DEFAULT_BLOCK, help="Block length (default 32)")
    p.add_argument("--gen-length", type=int, default=DEFAULT_GEN, help="Generation length (default 256)")
    p.add_argument("--skip-slow", action="store_true", help="Skip slow methods in unified benchmark")
    p.add_argument("--mock", action="store_true", help="Forward --mock to unified benchmark")
    p.add_argument(
        "--dataset-hf",
        default="eth-sri/json-mode-eval-extended",
        help="Document only; dataset is fixed inside run_unified_benchmark.load_jsonschema_dataset",
    )
    p.add_argument("--run-lave", action="store_true", help="Run vendor/dgrammar/bench/run_lave_timed.py after ours")
    p.add_argument("--skip-ours", action="store_true", help="Only run LAVE / aggregate (no SDSD run)")
    p.add_argument("--lave-seed", type=int, default=0, help="First argument to run_lave_timed.py")
    p.add_argument(
        "--lave-extra",
        type=int,
        default=0,
        help="Last argument to run_lave_timed.py (vendor convention; often 0)",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="After runs, call aggregate_unified_results.py on our dir + vendor LAVE results",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only run aggregate_unified_results.py (no benchmark subprocesses)",
    )
    p.add_argument(
        "--our-results",
        type=Path,
        default=None,
        help="With --aggregate-only: our JSONL directory (default: --output)",
    )
    p.add_argument(
        "--lave-results",
        type=Path,
        default=None,
        help="With --aggregate-only: LAVE JSONL directory (default: vendor/dgrammar/results)",
    )
    args = p.parse_args()

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = out_dir.resolve()

    _print_shared_config(
        steps=args.steps,
        block_length=args.block_length,
        gen_length=args.gen_length,
        dataset=args.dataset_hf,
        lave_seed=args.lave_seed,
    )

    if args.aggregate_only:
        our = args.our_results or out_dir
        our = our if our.is_absolute() else ROOT / our
        our = our.resolve()
        lave_r = args.lave_results or (ROOT / "vendor" / "dgrammar" / "results")
        if not isinstance(lave_r, Path):
            lave_r = Path(lave_r)
        if not lave_r.is_absolute():
            lave_r = ROOT / lave_r
        lave_r = lave_r.resolve()
        return _run_aggregate(our, lave_r if lave_r.is_dir() else None)

    rc = 0
    if not args.skip_ours:
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = _run_unified_benchmark_subprocess(args, out_dir)
        if rc != 0:
            print(f"run_unified_benchmark.py exited with {rc}")
            return rc

    if args.run_lave:
        rc_l = _run_lave_vendor(args)
        if rc_l != 0 and rc == 0:
            rc = rc_l

    if args.aggregate:
        lave_dir = ROOT / "vendor" / "dgrammar" / "results"
        ar = _run_aggregate(out_dir, lave_dir if lave_dir.is_dir() else None)
        if ar != 0 and rc == 0:
            rc = ar

    print(f"\nDone. Our JSONL: {out_dir}")
    if args.aggregate:
        print(f"Comparison JSON: {out_dir / 'unified_comparison.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

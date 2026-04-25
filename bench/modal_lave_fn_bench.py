"""Run LAVE false-negative detection on Modal A100.

The image uses ``add_local_file`` for ``run_lave_fn_detection.py`` from this repo's
``bench/`` directory each time the image layer is built (local file changes should
invalidate that layer). If results look stale, confirm JSONL has
``fn_detection_script_version`` and check Modal logs for the printed sha256.

For every block that LAVE's beam-search rejects, ``oracle_fast`` (default
``dfs``; configurable) checks whether ANY valid token assignment exists.
False negatives are logged without changing LAVE's validate outcome.

Outputs one JSONL per chunk:
  lave_fn_detection_{dataset}_s{seed}_t{steps}_oml{oml}[_off{offset}].jsonl

Quick test (10 instances, 1 chunk):
    modal run bench/modal_lave_fn_bench.py --total 10 --chunks 1

Full jsonschema run (272 instances, 2 chunks):
    modal run bench/modal_lave_fn_bench.py

Full ``jsb_hard`` (test split only, 368 instances — ``jsb_*`` defaults to test in ``jsb_dataset``):
    modal run bench/modal_lave_fn_bench.py --dataset jsb_hard --total 368 --chunks 2

oracle_mask_limit controls the max number of MASKs in a block before the oracle
is skipped (default 12; reduce to 8 for faster runs on complex grammars).

Oracle search mode (``oracle_fast``): pass
``--oracle-search-mode dfs|bfs|bfs_dedup|smart`` (default ``dfs``).
Sets ``DGRAMMAR_ORACLE_SEARCH_MODE`` in the worker. Mode ``smart`` runs
``bfs_dedup`` then sound ``dfs`` when dedup is False; when dedup is True it
skips DFS (see ``oracle_fast`` module doc). Pass ``--oracle-dedup-probe-tokens 'id,id,...'`` to set
``DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS`` in the worker (refines dedup fingerprint).

Optional: ``--oracle-string-prune-threshold N`` sets
``DGRAMMAR_ORACLE_STRING_PRUNE_THRESHOLD`` (MASK positions with >N allowed ids
only try one — **unsound**, faster on TN-heavy string branches).

Optional: ``--oracle-max-search-seconds S`` caps per-call block search time (default 3.0;
pass 0 to disable). Prevents wall-clock blowups on instances like heavy rejects × oracle.

Optional: ``--oracle-sample-rate P`` (default 1.0): on each LAVE reject, run the oracle
with probability ``P``. Use e.g. 0.1 to cut oracle wall time ~10× when estimating FN rate.

Optional: ``--oracle-trie-order`` sets ``DGRAMMAR_ORACLE_TRIE_ORDER=1`` (prefix trie over
``dbg_tokens`` for MASK candidate order).
"""

from pathlib import Path
from typing import Optional

import modal

_BENCH_DIR    = Path(__file__).resolve().parent
_DGRAMMAR_DIR = _BENCH_DIR.parent
_cd4d_candidates = (
    _DGRAMMAR_DIR / "vendors" / "CD4dLLM",
    _DGRAMMAR_DIR / "vendor"  / "CD4dLLM",
)
_CD4D_LLM = next((p for p in _cd4d_candidates if p.is_dir()), _cd4d_candidates[0])

app = modal.App("lave-fn-detection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "accelerate>=0.30",
        "numpy",
        "frozendict",
        "jsonschema",
        "datasets==2.21.0",
        "setuptools<75",
        "maturin",
        "llguidance>=1.6",
        "huggingface_hub",
        "stopit",
    )
    .add_local_dir(str(_CD4D_LLM), "/root/CD4dLLM", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/CD4dLLM/rustformlang_bindings && "
        "maturin build --release && "
        "pip install target/wheels/*cp312*.whl && "
        "cd /root/CD4dLLM && pip install -e .",
    )
    .add_local_file(
        str(_BENCH_DIR / "run_lave_fn_detection.py"),
        "/root/run_lave_fn_detection.py",
    )
    # Oracle: default dfs; override with --oracle-search-mode bfs|bfs_dedup|smart
    .add_local_file(str(_BENCH_DIR / "oracle_fast.py"), "/root/oracle_fast.py")
    .add_local_file(str(_BENCH_DIR / "jsb_dataset.py"), "/root/jsb_dataset.py")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(
    seed: int,
    limit: int,
    offset: int,
    steps: int,
    dataset: str = "jsonschema",
    instance_timeout: int = 120,
    oracle_mask_limit: int = 12,
    oracle_search_mode: str = "dfs",
    oracle_string_prune_threshold: Optional[int] = None,
    oracle_max_search_seconds: float = 3.0,
    oracle_sample_rate: float = 1.0,
    oracle_trie_order: int = 0,
    oracle_dedup_probe_tokens: Optional[str] = None,
):
    import hashlib
    import subprocess, shutil, os

    _rld = "/root/run_lave_fn_detection.py"
    with open(_rld, "rb") as _f:
        _h = hashlib.sha256(_f.read()).hexdigest()
    print(f"[fn_detection] {_rld} sha256={_h[:20]}... (compare with: shasum -a 256 bench/run_lave_fn_detection.py)")

    ds_safe  = dataset.replace("/", "_")
    sfx      = f"_off{offset}" if offset > 0 else ""
    oml_tag  = f"_oml{oracle_mask_limit}"
    local_fn = f"/root/results/lave_fn_detection_{ds_safe}_s{seed}_t{steps}{oml_tag}{sfx}.jsonl"
    out_fn   = f"/results/lave_fn_detection_{ds_safe}_s{seed}_t{steps}{oml_tag}{sfx}.jsonl"

    if os.path.exists(out_fn):
        os.remove(out_fn)

    sub_env = {
        "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/root",
        "PYTHONPATH": "/root:/root/CD4dLLM",
        "DGRAMMAR_ORACLE_SEARCH_MODE": oracle_search_mode,
    }
    if oracle_string_prune_threshold is not None:
        sub_env["DGRAMMAR_ORACLE_STRING_PRUNE_THRESHOLD"] = str(
            oracle_string_prune_threshold
        )
    if oracle_trie_order:
        sub_env["DGRAMMAR_ORACLE_TRIE_ORDER"] = "1"
    if oracle_dedup_probe_tokens:
        sub_env["DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS"] = oracle_dedup_probe_tokens

    result = subprocess.run(
        [
            "python", "/root/run_lave_fn_detection.py",
            str(seed), str(limit), dataset, str(steps),
            str(offset), str(instance_timeout), str(oracle_mask_limit),
            str(oracle_max_search_seconds),
            str(oracle_sample_rate),
        ],
        capture_output=True,
        text=True,
        cwd="/root",
        env=sub_env,
    )
    print(result.stdout[-5000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    try:
        shutil.copy2(local_fn, out_fn)
        RESULTS_VOL.commit()
        print(f"Saved to {out_fn}")
    except FileNotFoundError:
        print(f"Result file not found: {local_fn}")

    return result.stdout[-5000:] if result.stdout else result.stderr[-2000:]


@app.local_entrypoint()
def main(
    seed: int = 0,
    total: int = 272,
    steps: int = 128,
    chunks: int = 2,
    dataset: str = "jsonschema",
    instance_timeout: int = 120,
    oracle_mask_limit: int = 12,
    oracle_search_mode: str = "dfs",
    oracle_string_prune_threshold: Optional[int] = None,
    oracle_max_search_seconds: float = 3.0,
    oracle_sample_rate: float = 1.0,
    oracle_trie_order: int = 0,
    oracle_dedup_probe_tokens: Optional[str] = None,
):
    """
    Run LAVE with false-negative oracle on Modal A100s.

    Key metrics in the output JSONL:
      fn_summary.false_negatives  — blocks where LAVE rejected but oracle found valid assignment
      fn_summary.true_negatives   — blocks both LAVE and oracle rejected
      fn_summary.fn_rate          — false_negatives / (false_negatives + true_negatives)
      fn_summary.skipped_blocks   — oracle skipped (MASK limit, or search timeout if capped)
      fn_summary.oracle_sample_skips — rejects where oracle was not run (sample_rate < 1)
      fn_summary.oracle_sample_rate — configured P for Bernoulli sampling per reject
    """
    chunk_size = (total + chunks - 1) // chunks

    handles = []
    for i in range(chunks):
        off   = i * chunk_size
        limit = min(chunk_size, total - off)
        if limit <= 0:
            break
        print(
            f"Chunk {i}: offset={off}, limit={limit}, dataset={dataset}, "
            f"seed={seed}, T={steps}, oracle_mask_limit={oracle_mask_limit}, "
            f"oracle_search_mode={oracle_search_mode}, "
            f"oracle_string_prune_threshold={oracle_string_prune_threshold}, "
            f"oracle_max_search_seconds={oracle_max_search_seconds}, "
            f"oracle_sample_rate={oracle_sample_rate}, "
            f"oracle_trie_order={oracle_trie_order}, "
            f"oracle_dedup_probe_tokens={oracle_dedup_probe_tokens!r}"
        )
        h = run_chunk.spawn(
            seed,
            limit,
            off,
            steps,
            dataset,
            instance_timeout,
            oracle_mask_limit,
            oracle_search_mode,
            oracle_string_prune_threshold,
            oracle_max_search_seconds,
            oracle_sample_rate,
            oracle_trie_order,
            oracle_dedup_probe_tokens,
        )
        handles.append((i, h))

    for chunk_i, h in handles:
        result = h.get()
        print(f"\n{'='*60}")
        print(f"=== Chunk {chunk_i} ===")
        print(f"{'='*60}")
        print(result)

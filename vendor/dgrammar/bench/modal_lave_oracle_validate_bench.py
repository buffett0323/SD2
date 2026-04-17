"""Run ``run_lave_oracle_validate.py`` (line-2 oracle replace) on Modal A100.

Same worker setup as ``modal_lave_fn_bench.py``, but executes ``run_lave_oracle_validate.py``.
Outputs: ``lave_oracle_validate_{dataset}_s{seed}_t{steps}_oml{oml}...jsonl``

``intercepted_validate`` passes LAVE's ``p`` tensor (softmax over the block positions) into
``oracle_find_block_assignment`` as ``model_logits`` with ``logits_index_base=index_to_consume``.
``oracle_fast`` then orders MASK branches by descending score at each position — same signal
LAVE uses in beam search — which greatly cuts DFS dead-ends on false-negative-style blocks.

This worker sets ``DGRAMMAR_ORACLE_LOGITS_PRIORITY=1`` explicitly so remote runs always use
that ordering (even if defaults change). Disable with ``--no-oracle-logits-priority``.

Wall clock: oracle-validate can exceed plain FN detection because it changes generation; defaults
use a longer Modal function timeout and per-instance SIGALRM budget than the FN bench.

Default ``oracle_search_mode=smart``: ``oracle_find_block_assignment`` runs dedup + sound DFS
for unsat, then assignment DFS only when the block may be satisfiable (cuts many TN timeouts vs
plain ``dfs``). Override with ``--oracle-search-mode dfs`` if you want the single-pass DFS only.

Optional ``--oracle-dedup-probe-tokens 'id,id,...'`` sets ``DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS``.

Example::

    modal run bench/modal_lave_oracle_validate_bench.py --total 10 --chunks 1 --dataset jsb_medium

Full run with more headroom::

    modal run bench/modal_lave_oracle_validate_bench.py --chunks 4 --instance-timeout 600
"""

from pathlib import Path
from typing import Optional

import modal

_BENCH_DIR = Path(__file__).resolve().parent
_DGRAMMAR_DIR = _BENCH_DIR.parent
_cd4d_candidates = (
    _DGRAMMAR_DIR / "vendors" / "CD4dLLM",
    _DGRAMMAR_DIR / "vendor" / "CD4dLLM",
)
_CD4D_LLM = next((p for p in _cd4d_candidates if p.is_dir()), _cd4d_candidates[0])

app = modal.App("lave-oracle-validate")

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
        str(_BENCH_DIR / "run_lave_oracle_validate.py"),
        "/root/run_lave_oracle_validate.py",
    )
    .add_local_file(str(_BENCH_DIR / "oracle_fast.py"), "/root/oracle_fast.py")
    .add_local_file(str(_BENCH_DIR / "jsb_dataset.py"), "/root/jsb_dataset.py")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=14_400,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(
    seed: int,
    limit: int,
    offset: int,
    steps: int,
    dataset: str = "jsonschema",
    instance_timeout: int = 300,
    oracle_mask_limit: int = 12,
    oracle_search_mode: str = "smart",
    oracle_string_prune_threshold: Optional[int] = None,
    oracle_max_search_seconds: float = 3.0,
    oracle_sample_rate: float = 1.0,
    oracle_trie_order: int = 0,
    oracle_logits_priority: bool = True,
    oracle_dedup_probe_tokens: Optional[str] = None,
    line2_debug: bool = False,
    oracle_call_mask_limit: int = 0,
    oracle_no_rescue_budget: int = 0,
):
    import hashlib
    import subprocess, shutil, os

    _scr = "/root/run_lave_oracle_validate.py"
    with open(_scr, "rb") as _f:
        _h = hashlib.sha256(_f.read()).hexdigest()
    print(f"[oracle_validate] {_scr} sha256={_h[:20]}...")

    ds_safe = dataset.replace("/", "_")
    sfx = f"_off{offset}" if offset > 0 else ""
    oml_tag = f"_oml{oracle_mask_limit}"
    local_fn = f"/root/results/lave_oracle_validate_{ds_safe}_s{seed}_t{steps}{oml_tag}{sfx}.jsonl"
    out_fn = f"/results/lave_oracle_validate_{ds_safe}_s{seed}_t{steps}{oml_tag}{sfx}.jsonl"

    if os.path.exists(out_fn):
        os.remove(out_fn)

    sub_env = {
        "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/root",
        "PYTHONPATH": "/root:/root/CD4dLLM",
        "DGRAMMAR_ORACLE_SEARCH_MODE": oracle_search_mode,
        "DGRAMMAR_ORACLE_LOGITS_PRIORITY": "1" if oracle_logits_priority else "0",
    }
    if oracle_string_prune_threshold is not None:
        sub_env["DGRAMMAR_ORACLE_STRING_PRUNE_THRESHOLD"] = str(
            oracle_string_prune_threshold
        )
    if oracle_trie_order:
        sub_env["DGRAMMAR_ORACLE_TRIE_ORDER"] = "1"
    if oracle_dedup_probe_tokens:
        sub_env["DGRAMMAR_ORACLE_DEDUP_PROBE_TOKENS"] = oracle_dedup_probe_tokens
    if line2_debug:
        sub_env["DGRAMMAR_LINE2_DEBUG"] = "1"

    result = subprocess.run(
        [
            "python",
            "/root/run_lave_oracle_validate.py",
            str(seed),
            str(limit),
            dataset,
            str(steps),
            str(offset),
            str(instance_timeout),
            str(oracle_mask_limit),
            str(oracle_max_search_seconds),
            str(oracle_sample_rate),
            str(oracle_call_mask_limit),
            str(oracle_no_rescue_budget),
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
    instance_timeout: int = 300,
    oracle_mask_limit: int = 12,
    oracle_search_mode: str = "smart",
    oracle_string_prune_threshold: Optional[int] = None,
    oracle_max_search_seconds: float = 3.0,
    oracle_sample_rate: float = 1.0,
    oracle_trie_order: int = 0,
    no_oracle_logits_priority: bool = False,
    oracle_dedup_probe_tokens: Optional[str] = None,
    line2_debug: bool = False,
    oracle_call_mask_limit: int = 0,
    oracle_no_rescue_budget: int = 0,
):
    chunk_size = (total + chunks - 1) // chunks
    handles = []
    for i in range(chunks):
        off = i * chunk_size
        limit = min(chunk_size, total - off)
        if limit <= 0:
            break
        print(
            f"Chunk {i}: offset={off}, limit={limit}, dataset={dataset}, "
            f"instance_timeout={instance_timeout}, "
            f"oracle_max_search_seconds={oracle_max_search_seconds}, "
            f"oracle_logits_priority={not no_oracle_logits_priority}, "
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
            not no_oracle_logits_priority,
            oracle_dedup_probe_tokens,
            line2_debug,
            oracle_call_mask_limit,
            oracle_no_rescue_budget,
        )
        handles.append((i, h))

    for chunk_i, h in handles:
        out = h.get()
        print(f"\n{'='*60}\n=== Chunk {chunk_i} ===\n{'='*60}\n{out}")

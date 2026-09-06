"""Run Dgrammar async overlap (llguidance) with timing on Modal A100."""

from pathlib import Path

import modal

# Absolute paths derived from this file's location so `modal run` works from any cwd.
_BENCH_DIR   = Path(__file__).resolve().parent
_DGRAMMAR_DIR = _BENCH_DIR.parent
_cd4d_candidates = (
    _DGRAMMAR_DIR / "vendors" / "CD4dLLM",
    _DGRAMMAR_DIR / "vendor"  / "CD4dLLM",
)
_CD4D_LLM = next((p for p in _cd4d_candidates if p.is_dir()), _cd4d_candidates[0])

app = modal.App("v2-async-timed-bench")

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
    )
    .add_local_dir(str(_CD4D_LLM), "/root/constrained-diffusion", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "rm -rf target/wheels && "
        "maturin build --release && "
        "pip install target/wheels/*.whl && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
    .add_local_dir(str(_DGRAMMAR_DIR / "dgrammar"), "/root/dgrammar", copy=True)
    .add_local_file(str(_BENCH_DIR / "run_dgrammar_timed.py"), "/root/run_dgrammar_timed.py")
    .add_local_file(str(_BENCH_DIR / "jsb_dataset.py"), "/root/jsb_dataset.py")
    .add_local_file(str(_DGRAMMAR_DIR / "pyproject.toml"), "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    # A100-80GB.  On jsb_hard the schema pushes prompts past 5k tokens and the
    # confidence softmax is taken in fp64, so a 40GB card OOMs part-way: 105 of
    # 269 DPGrammar instances and 63 of 269 no-DP ones died that way, and
    # because the DP layer holds more state it was penalised harder than the
    # baseline it is compared against.  The margin that produced was an
    # artifact of unequal OOM rates, not of repair.
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(seed: int, limit: int, offset: int, steps: int, block_ar: int = 1,
              dataset: str = "jsonschema", method: str = "dgrammar",
              instance_ids: str = "", tag: str = "",
              gen_length: int = 256,
              oracle: bool = False, window_mode: str = "full",
              objective: str = "logp", cand_source: str = "automaton",
              eos_in_candidates: bool = False,
              top_k_dp: int = 100, max_lookahead: int = 48,
              remasking: str = "low_confidence"):
    import subprocess
    import shutil
    import os

    method_tag = "dp" if method == "dp" else ("v2_async_ac4_timed" if block_ar else "v2_async_ac4_fullpar_timed")
    ds_safe = dataset.replace("/", "_")
    suffix = f"_off{offset}" if offset > 0 else ""
    tag_sfx = f"_{tag}" if tag else ""
    local_file = f"/root/results/{method_tag}_{ds_safe}_s{seed}_t{steps}{suffix}{tag_sfx}.jsonl"
    out_file = f"/results/{method_tag}_{ds_safe}_s{seed}_t{steps}{suffix}{tag_sfx}.jsonl"

    if os.path.exists(out_file):
        os.remove(out_file)

    cmd = [
        "python", "/root/run_dgrammar_timed.py",
        str(seed), str(limit), dataset, str(steps), str(offset),
        str(block_ar), method,
        instance_ids,           # argv[8]: may be empty string
        tag,                      # argv[9]:  optional filename tag suffix
        str(gen_length),          # argv[10]: generation length (default 256)
        "1" if oracle else "0",   # argv[11]: run the unfrozen-prefix oracle probe
        window_mode,              # argv[12]: "full", "progressive", "descend", "wN"
        objective,                # argv[13]: "logp" or "min_edit"
        cand_source,              # argv[14]: "automaton" or "vocab"
        "1" if eos_in_candidates else "0",   # argv[15]
        str(top_k_dp),            # argv[16]: candidate budget k per lattice node
        str(max_lookahead),       # argv[17]: span budget L, the forward-scan cap
        remasking,                # argv[18]: which masked position a step reveals
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/root",
        env={
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "PYTHONPATH": "/root:/root/constrained-diffusion",
        },
    )
    print(result.stdout[-5000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    try:
        shutil.copy2(local_file, out_file)
        print(f"Saved to {out_file}")
    except FileNotFoundError:
        print(f"Result file not found: {local_file}")

    return result.stdout[-5000:] if result.stdout else result.stderr[-2000:]


@app.local_entrypoint()
def main(
    seed: int = 0,
    total: int = 272,
    steps: int = 128,
    chunks: int = 2,
    block_ar: int = 1,
    dataset: str = "jsonschema",
    method: str = "dgrammar",
    instance_ids: str = "",
    tag: str = "",
    gen_length: int = 256,
    oracle: bool = False,
    window_mode: str = "full",
    objective: str = "logp",
    cand_source: str = "automaton",
    eos_in_candidates: bool = False,
    top_k_dp: int = 100,
    max_lookahead: int = 48,
    remasking: str = "low_confidence",
):
    """
    --method dgrammar   original greedy violation-retry (default)
    --method dp         DP-based global grammar correction (generate_dp)
    --instance-ids IDS  comma-separated instance IDs to run only those
                        (e.g. --instance-ids o33928,o12618,o70379)
                        When set, runs as a single chunk ignoring total/chunks.
    --tag TAG           append _TAG to the output filename to avoid overwriting
                        the main results (e.g. --tag debug)
    --gen-length N      generation length in tokens (default 256; use 512 for
                        schemas that need longer output)
    --oracle            at each repair site, re-run the DP with the prefix
                        unfrozen and record whether the better answer was
                        reachable at all. Measurement only; the generated output
                        is identical with or without it.
    --objective {logp,min_edit}
                        `logp` maximises summed log-probability over the repair
                        span, which rewrites positions the grammar never
                        objected to -- 5.95 of a 14-position span on a pilot,
                        where only the violator was illegal.  `min_edit` orders
                        candidates by how many of the model's tokens they change
                        first, so the repair touches only what the grammar
                        forces.
    --remasking {low_confidence,random}
                        which masked position a step reveals. `low_confidence`
                        is the reported schedule; any other value picks at
                        random, which tests schedule-agnosticism.
    --top-k-dp K        candidate budget per lattice node (default 100, the
                        reported configuration)
    --max-lookahead L   span budget: how far the forward scan may run before it
                        gives up on finding a depth-zero junction (default 48)
    --window-mode {full,progressive,descend,wN}
                        `full` (default) hands the DP the whole constraint span
                        in one shot.  `progressive` is the published DPGrammar
                        order -- it doubles the span and stops at the first width
                        admitting any valid path, which on a pilot was width 1 at
                        every site, so the DP changed one token and a closing
                        token there ends the document.  Kept as the baseline arm.
    """
    if instance_ids:
        ids_list = instance_ids.split(",")
        print(f"Running [{method}] on 1x A100: {dataset}, seed={seed}, T={steps}, gen_length={gen_length}, tag={tag!r}")
        print(f"Instance filter: {ids_list}")
        handle = run_chunk.spawn(seed, len(ids_list), 0, steps, block_ar, dataset, method, instance_ids, tag, gen_length, oracle, window_mode, objective, cand_source, eos_in_candidates, top_k_dp, max_lookahead, remasking)
        result = handle.get()
        print(result)
        return

    chunk_size = (total + chunks - 1) // chunks
    mode = "block_ar=32" if block_ar else "full_parallel=256"
    print(f"Running [{method}] on {chunks}x A100: {dataset}, seed={seed}, T={steps}, gen_length={gen_length}, {mode}, tag={tag!r}")
    print(f"Total={total}, chunk_size={chunk_size}")

    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={offset}, limit={limit}")
        handles.append(run_chunk.spawn(seed, limit, offset, steps, block_ar, dataset, method, "", tag, gen_length, oracle, window_mode, objective, cand_source, eos_in_candidates, top_k_dp, max_lookahead, remasking))

    for i, handle in enumerate(handles):
        result = handle.get()
        print(f"\n{'='*60}")
        print(f"=== Chunk {i} ===")
        print(f"{'='*60}")
        print(result)

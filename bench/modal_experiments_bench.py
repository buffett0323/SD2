"""Modal launcher for all dgrammar paper experiments (A, B, C, D).

Usage examples:

  # Experiment A — top-k coverage (50 instances)
  modal run bench/modal_experiments_bench.py --exp A --total 50 --chunks 1

  # Experiment B — batch size ablation {1,2,4,8}
  modal run bench/modal_experiments_bench.py --exp B --total 50 --chunks 1

  # Experiment C — async overlap timing (reuses run_dgrammar_timed.py)
  modal run bench/modal_experiments_bench.py --exp C --total 272 --chunks 4

  # Experiment D — dgrammar vs unconstrained validity comparison
  modal run bench/modal_experiments_bench.py --exp D --total 272 --chunks 4

  # Run all experiments in one call (sequential per chunk, parallel chunks)
  modal run bench/modal_experiments_bench.py --exp ALL --total 50 --chunks 1
"""

from pathlib import Path
from typing import Optional

import modal

_BENCH_DIR  = Path(__file__).resolve().parent
_DGRAMMAR_DIR = _BENCH_DIR.parent
_cd4d_candidates = (
    _DGRAMMAR_DIR / "vendors" / "CD4dLLM",
    _DGRAMMAR_DIR / "vendor"  / "CD4dLLM",
)
_CD4D_LLM = next((p for p in _cd4d_candidates if p.is_dir()), _cd4d_candidates[0])

app = modal.App("dgrammar-experiments")

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
    # dgrammar package
    .add_local_dir(str(_DGRAMMAR_DIR / "dgrammar"), "/root/dgrammar", copy=True)
    # bench scripts
    .add_local_file(str(_BENCH_DIR / "run_experiments.py"),    "/root/run_experiments.py")
    .add_local_file(str(_BENCH_DIR / "run_dgrammar_timed.py"), "/root/run_dgrammar_timed.py")
    .add_local_file(str(_BENCH_DIR / "jsb_dataset.py"),        "/root/jsb_dataset.py")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)

_ENV = {
    "PATH":                        "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
    "HOME":                        "/root",
    "PYTHONPATH":                  "/root:/root/CD4dLLM",
    "PYTORCH_CUDA_ALLOC_CONF":     "expandable_segments:True",
}


@app.function(
    image=image,
    gpu="A100",
    timeout=14_400,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(
    exp: str,
    seed: int,
    limit: int,
    offset: int,
    steps: int,
    dataset: str = "jsb_medium",
    batch_sizes: str = "1,2,4,8",   # for Exp B
):
    import subprocess, shutil, os, hashlib

    # Log script hash for reproducibility
    for scr in ["/root/run_experiments.py", "/root/run_dgrammar_timed.py"]:
        if os.path.exists(scr):
            h = hashlib.sha256(open(scr, "rb").read()).hexdigest()
            print(f"[hash] {scr} = {h[:20]}...")

    ds_safe = dataset.replace("/", "_")
    sfx     = f"_off{offset}" if offset > 0 else ""
    exps    = ["A", "B", "C", "D"] if exp == "ALL" else [exp.upper()]

    for e in exps:
        local_fn = f"/root/results/exp_{e}_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"
        out_fn   = f"/results/exp_{e}_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"

        if os.path.exists(out_fn):
            os.remove(out_fn)

        cmd = [
            "python", "/root/run_experiments.py",
            e, str(seed), str(limit), dataset, str(steps), str(offset),
        ]
        if e == "B":
            cmd.append(batch_sizes)

        print(f"\n{'='*50}\nRunning Exp {e}: {' '.join(cmd[2:])}\n{'='*50}")
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd="/root", env=_ENV,
        )
        print(result.stdout[-8000:] if result.stdout else "")
        if result.stderr:
            print("STDERR:", result.stderr[-2000:])

        # For Exp C, run_dgrammar_timed.py writes to a different path — copy both
        if e == "C":
            timed_fn = f"/root/results/v2_async_ac4_timed_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"
            timed_out = f"/results/v2_async_ac4_timed_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"
            try:
                shutil.copy2(timed_fn, timed_out)
                RESULTS_VOL.commit()
                print(f"Saved Exp C to {timed_out}")
            except FileNotFoundError:
                print(f"Exp C result not found: {timed_fn}")
        else:
            try:
                shutil.copy2(local_fn, out_fn)
                RESULTS_VOL.commit()
                print(f"Saved Exp {e} to {out_fn}")
            except FileNotFoundError:
                print(f"Exp {e} result not found: {local_fn}")

    return result.stdout[-5000:] if result.stdout else result.stderr[-2000:]


@app.local_entrypoint()
def main(
    exp:      str = "C",
    seed:     int = 0,
    total:    int = 50,
    steps:    int = 128,
    chunks:   int = 2,
    dataset:  str = "jsb_medium",
    batch_sizes: str = "1,2,4,8",
):
    chunk_size = (total + chunks - 1) // chunks
    exps = ["A", "B", "C", "D"] if exp.upper() == "ALL" else [exp.upper()]

    print(f"dgrammar experiments: {exps}")
    print(f"dataset={dataset}, seed={seed}, T={steps}, total={total}, chunks={chunks}")
    if "B" in exps:
        print(f"  Exp B batch_sizes={batch_sizes}")

    handles = []
    for i in range(chunks):
        off   = i * chunk_size
        limit = min(chunk_size, total - off)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={off}, limit={limit}")
        h = run_chunk.spawn(exp.upper(), seed, limit, off, steps, dataset, batch_sizes)
        handles.append((i, h))

    for chunk_i, h in handles:
        out = h.get()
        print(f"\n{'='*60}\n=== Chunk {chunk_i} ===\n{'='*60}\n{out}")

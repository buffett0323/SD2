"""Run IG-CD (ETH baseline) with timing instrumentation on Modal A100."""

import modal

app = modal.App("igcd-timed-bench")

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
        "huggingface_hub",
        "stopit",
    )
    .add_local_dir("../vendor/constrained-diffusion", "/root/constrained-diffusion", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "maturin build --release && "
        "pip install target/wheels/*.whl && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
    .add_local_file("run_igcd_timed.py", "/root/run_igcd_timed.py")
    .add_local_file("jsb_dataset.py", "/root/jsb_dataset.py")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(seed: int, limit: int, offset: int, steps: int,
              dataset: str = "jsonschema", instance_timeout: int = 120):
    import subprocess
    import shutil
    import os

    ds_safe = dataset.replace("/", "_")
    suffix = f"_off{offset}" if offset > 0 else ""
    local_file = f"/root/results/igcd_timed_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"
    out_file = f"/results/igcd_timed_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"

    if os.path.exists(out_file):
        os.remove(out_file)

    result = subprocess.run(
        [
            "python", "/root/run_igcd_timed.py",
            str(seed), str(limit), dataset, str(steps), str(offset),
            str(instance_timeout),
        ],
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
    dataset: str = "jsonschema",
    instance_timeout: int = 120,
):
    chunk_size = (total + chunks - 1) // chunks
    print(f"Running IG-CD timed on {chunks}x A100: {dataset}, seed={seed}, T={steps}, timeout={instance_timeout}s")
    print(f"Total={total}, chunk_size={chunk_size}")

    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={offset}, limit={limit}")
        handles.append(run_chunk.spawn(seed, limit, offset, steps, dataset, instance_timeout))

    for i, handle in enumerate(handles):
        result = handle.get()
        print(f"\n{'='*60}")
        print(f"=== Chunk {i} ===")
        print(f"{'='*60}")
        print(result)

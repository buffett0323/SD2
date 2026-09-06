"""Modal runner for bench/merge_probe.py -- does key-merging discard the optimum?

CPU only: the probe reconstructs parser states by replaying generated outputs
through llguidance and plays the model's part with random score vectors, so no
GPU and no checkpoint are involved.  Chunks fan out across containers.

    modal run bench/modal_merge_probe.py                       # 24-instance pilot
    modal run bench/modal_merge_probe.py --instances 511 --chunks 16 --seeds 5
    modal volume get merge-probe-results merge_probe_off0.jsonl .
"""
from pathlib import Path

import modal

_BENCH_DIR = Path(__file__).resolve().parent
_ROOT = _BENCH_DIR.parent

app = modal.App("dpgrammar-merge-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "jsonschema", "llguidance>=1.7", "huggingface_hub")
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    )
)

# This module is imported twice: locally to build and launch, and again inside
# the container when it starts.  Everything that reads the local filesystem has
# to be confined to the first, or the container import fails on paths that only
# exist on the developer's machine.  The container does not need these calls
# anyway -- by then the image they describe is already built.
if modal.is_local():
    # Only the run whose outputs are replayed is shipped (~3MB), not results/.
    _SOURCE_RUNS = sorted((_ROOT / "results").glob("dp_jsb_medium_s0_t128*v6dp.jsonl"))
    assert _SOURCE_RUNS, f"no v6dp result shards under {_ROOT / 'results'}"
    image = (
        image
        .add_local_dir(str(_ROOT / "dgrammar"), "/root/dgrammar", copy=True)
        .add_local_file(str(_BENCH_DIR / "merge_probe.py"),
                        "/root/bench/merge_probe.py", copy=True)
    )
    for _f in _SOURCE_RUNS:
        image = image.add_local_file(str(_f), f"/root/results/{_f.name}", copy=True)

RESULTS_VOL = modal.Volume.from_name("merge-probe-results", create_if_missing=True)


# ~7.8 s per trial measured locally, so a 32-instance chunk at 12 sites and 3
# seeds runs a little over two hours.  The old 7200 s ceiling cut that off.
@app.function(image=image, cpu=4.0, memory=16384, timeout=21600,
              volumes={"/out": RESULTS_VOL})
def run_chunk(offset: int, instances: int, span: int, stride: int, top_k: int,
              beams: str, seeds: int, max_live: int, max_sites: int, tag: str):
    import os
    import subprocess
    import time

    name = f"merge_probe{('_' + tag) if tag else ''}_off{offset}.jsonl"
    local = f"/root/{name}"
    out = f"/out/{name}"

    # merge_probe.py flushes after every instance, so pushing the partial file
    # to the volume as the run goes means a preempted or timed-out container
    # still contributes every trial it finished.  The push never shortens a
    # stored result, so a restarted attempt cannot truncate an earlier one.
    def _flush() -> None:
        try:
            if not os.path.exists(local):
                return
            with open(local) as fh:
                lines = fh.readlines()
            if lines and not lines[-1].endswith("\n"):
                lines = lines[:-1]
            if not lines:
                return
            prev = 0
            if os.path.exists(out):
                with open(out) as fh:
                    prev = sum(1 for _ in fh)
            if len(lines) <= prev:
                return
            with open(out, "w") as fh:
                fh.writelines(lines)
            RESULTS_VOL.commit()
            print(f"[flush] {len(lines)} trials saved", flush=True)
        except Exception as exc:
            print(f"[flush] failed: {type(exc).__name__}: {exc}", flush=True)

    cmd = [
        "python", "/root/bench/merge_probe.py",
        "--results", "/root/results/dp_jsb_medium_s0_t128*v6dp.jsonl",
        "--instances", str(instances), "--offset", str(offset),
        "--span", str(span), "--stride", str(stride),
        "--top-k", str(top_k), "--beams", beams, "--seeds", str(seeds),
        "--max-live", str(max_live),
        "--max-sites-per-instance", str(max_sites),
        "--out", local,
    ]
    log_path = "/root/merge_probe.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                text=True, cwd="/root",
                                env={"PATH": "/usr/local/bin:/usr/bin:/bin",
                                     "HOME": "/root", "PYTHONPATH": "/root"})
        while proc.poll() is None:
            time.sleep(30)
            _flush()
    proc.wait()
    _flush()

    try:
        with open(log_path) as fh:
            log = fh.read()
    except OSError:
        log = ""
    print(log[-8000:])
    if proc.returncode != 0:
        print(f"merge_probe.py exited {proc.returncode}")
    return log[-8000:]


@app.local_entrypoint()
def main(instances: int = 24, chunks: int = 1, span: int = 8, stride: int = 7,
         top_k: int = 100, beams: str = "1,2,4", seeds: int = 3,
         max_live: int = 2048, max_sites: int = 24, tag: str = ""):
    per = (instances + chunks - 1) // chunks
    handles = []
    for i in range(chunks):
        off = i * per
        n = min(per, instances - off)
        if n <= 0:
            break
        print(f"chunk {i}: offset={off} instances={n}")
        handles.append(run_chunk.spawn(off, n, span, stride, top_k, beams,
                                       seeds, max_live, max_sites, tag))
    for i, h in enumerate(handles):
        print(f"\n{'='*60}\n=== chunk {i} ===\n{'='*60}")
        print(h.get())

"""Launch proactively aligned FA decoding on Modal A100s.

Differences from ``modal_fa_bench``'s sibling ``modal_dgrammar_bench.py``:

  * The constraint comes from ``outlines-core``, not rustformlang. rustformlang
    is still built into the image because ``eval/dllm/model.py`` imports it
    transitively; it plays no part in constructing the automaton.
  * ``--decoder`` selects the mode/mass axis, and ``--sweep`` runs both halves
    of the 2x2 in one command.

Everything else mirrors ``modal_dgrammar_bench.py`` deliberately -- same base
image, same pins, same vendor install, same ``prepare_prompt``. Prompt
construction has to be identical to the baselines or the comparison means
nothing.

Usage
-----
    modal run bench/modal_fa_bench.py --task json --decoder marginal
    modal run bench/modal_fa_bench.py --task json --sweep          # both decoders
    modal run bench/modal_fa_bench.py --task smiles --max-depth 3 --chunks 1
    modal run bench/modal_fa_bench.py --task jsb_medium --sweep --chunks 4

Results land in the ``dgrammar-results`` volume as
``fa_<decoder>_<task>_s<seed>_t<steps>[_off<offset>][_<tag>].jsonl`` -- the same
schema ``bench/measure_degeneracy.py`` reads.
"""

from pathlib import Path

import modal

_BENCH_DIR = Path(__file__).resolve().parent
_ROOT = _BENCH_DIR.parent
_cd_candidates = (
    _ROOT / "vendor" / "constrained-diffusion",
    _ROOT / "vendors" / "constrained-diffusion",
)
_CD = next((p for p in _cd_candidates if p.is_dir()), _cd_candidates[0])

app = modal.App("fa-proactive-bench")

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
        # The FA path's own additions.
        "outlines-core>=0.2.14",   # token automaton; replaces the llguidance path
        "rdkit",                   # SMILES validity; without it that metric degrades
    )
    .add_local_dir(str(_CD), "/root/constrained-diffusion", copy=True,
                   ignore=["**/__pycache__", "**/target", "**/.git"])
    # rustformlang is NOT used to build the constraint here -- outlines-core is.
    # It is still required at import time: `eval/dllm/model.py` pulls in
    # `models/{llada,dream}/model.py`, which import `rustformlang.cfg` at module
    # level. Building it is cheaper than reimplementing `prepare_prompt`, and
    # reusing the vendor's prompt construction is what keeps this comparable to
    # the dgrammar/LAVE baselines.
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "rm -rf target/wheels && "
        "maturin build --release && "
        "pip install target/wheels/*.whl && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
    # Overlaid after the editable install so it wins: this repo's `eval/dllm` is
    # a fork of the vendor's that adds the JSONSchemaBench loader and the
    # humanevalpack fallback the CPP loader needs.
    .add_local_dir(str(_ROOT / "constrained_diffusion" / "eval" / "dllm"),
                   "/root/constrained-diffusion/constrained_diffusion/eval/dllm",
                   copy=True, ignore=["**/__pycache__"])
    .add_local_dir(str(_ROOT / "dgrammar"), "/root/dgrammar", copy=True,
                   ignore=["**/__pycache__"])
    .add_local_file(str(_BENCH_DIR / "run_fa_timed.py"), "/root/run_fa_timed.py")
    .add_local_file(str(_BENCH_DIR / "fa_tasks.py"), "/root/fa_tasks.py")
    .add_local_file(str(_BENCH_DIR / "jsb_dataset.py"), "/root/jsb_dataset.py")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


def _result_name(decoder, task, seed, steps, offset, tag):
    sfx = f"_off{offset}" if offset > 0 else ""
    tag_sfx = f"_{tag}" if tag else ""
    return f"fa_{decoder}_{task}_s{seed}_t{steps}{sfx}{tag_sfx}.jsonl"


@app.function(
    image=image,
    gpu="A100",
    # The 8B model plus the message-passing buffers overran the default
    # allocation on jsb_medium, where automata reach millions of edges.
    memory=32768,
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(task: str, decoder: str, seed: int, limit: int, offset: int,
              steps: int, gen_length: int, max_depth: int, max_edges: int,
              temperature: float, instance_ids: str, tag: str):
    import json
    import os
    import subprocess
    import time

    name = _result_name(decoder, task, seed, steps, offset, tag)
    local_file = f"/root/results/{name}"
    out_file = f"/results/{name}"

    # A100 capacity is scarce enough that these containers get preempted, and a
    # preempted container restarts its chunk from the first instance.  Copying
    # the result file only after the subprocess returned therefore threw away
    # every finished instance.  Instead the partial file is pushed to the volume
    # while the run is in flight, so a preemption costs one flush interval
    # rather than the whole chunk.
    #
    # The push is a union keyed on instance_id, stored rows first and this run's
    # rows overwriting them, so a partial attempt adds to what is already saved
    # instead of replacing it.  That makes three things safe: a restart after
    # preemption, a top-up of a chunk that died early, and an --instance-ids run
    # covering only the instances a previous attempt missed.  It never shrinks
    # the stored file.  Use a fresh --tag for a genuinely new run.
    def _flush() -> None:
        try:
            rows: dict[str, str] = {}
            for path in (out_file, local_file):       # local wins on conflict
                if not os.path.exists(path):
                    continue
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            iid = json.loads(line).get("instance_id")
                        except json.JSONDecodeError:
                            continue              # a half-written record
                        if iid:
                            rows[iid] = line
            if not rows:
                return
            prev = 0
            if os.path.exists(out_file):
                with open(out_file) as fh:
                    prev = sum(1 for line in fh if line.strip())
            if len(rows) < prev:                  # cannot happen, but never shrink
                return
            with open(out_file, "w") as fh:
                fh.write("\n".join(rows.values()) + "\n")
            RESULTS_VOL.commit()
            print(f"[flush] {len(rows)} instances in {out_file} "
                  f"(was {prev})", flush=True)
        except Exception as exc:                      # never kill the run over a flush
            print(f"[flush] failed: {type(exc).__name__}: {exc}", flush=True)

    cmd = [
        "python", "/root/run_fa_timed.py",
        "--task", task,
        "--decoder", decoder,
        "--seed", str(seed),
        "--limit", str(limit),
        "--offset", str(offset),
        "--steps", str(steps),
        "--gen-length", str(gen_length),
        "--max-depth", str(max_depth),
        "--max-edges", str(max_edges),
        "--temperature", str(temperature),
    ]
    if instance_ids:
        cmd += ["--instance-ids", instance_ids]
    if tag:
        cmd += ["--tag", tag]

    # Output goes to a file rather than a pipe: with Popen a full pipe buffer
    # would deadlock the poll loop below.
    log_path = "/root/run_fa.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, cwd="/root",
            env={
                "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/root",
                "PYTHONPATH": "/root:/root/constrained-diffusion",
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            },
        )
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
        print(f"run_fa_timed.py exited {proc.returncode}")
    if not os.path.exists(out_file):
        print(f"nothing saved: {local_file} was never written")
    return log[-8000:]


# Dataset sizes, used only to pick a sensible default --total.
_DEFAULT_TOTAL = {"json": 272, "smiles": 167, "cpp": 164, "jsb_medium": 586}


@app.local_entrypoint()
def main(
    task: str = "json",
    decoder: str = "marginal",
    sweep: bool = False,
    seed: int = 0,
    total: int = 0,
    chunks: int = 2,
    steps: int = 128,
    gen_length: int = 256,
    max_depth: int = 3,
    max_edges: int = 20_000_000,
    temperature: float = 1.0,
    instance_ids: str = "",
    tag: str = "",
    nowait: bool = False,
):
    """
    --task {json,smiles,cpp,jsb_medium}
                              benchmark to run (cpp is still blocked in fa_tasks).
                              jsb_medium is the degeneracy venue: it is where
                              DPGrammar was measured and where the existing
                              baselines live, but it has no reference outputs,
                              so judge it with bench/measure_degeneracy.py.
    --decoder {marginal,viterbi,sample}
                              mass vs mode. This is the axis under study, so a
                              result is only interpretable next to its pair.
    --sweep                   run viterbi and marginal together, which is the
                              proactive half of the 2x2 in one command
    --max-depth N             branch-nesting bound for non-regular grammars.
                              SMILES needs 3 for 100% reference coverage.
    --max-edges N             skip instances whose automaton exceeds this. Per
                              step cost is O(L*E) and the JSON tail reaches
                              ~20M edges, so leaving this unbounded will stall.
    --nowait                  spawn and return instead of streaming logs. Pair
                              with `modal run --detach` for a run that survives
                              closing the terminal; results still land in the
                              volume either way.
    """
    if total <= 0:
        total = _DEFAULT_TOTAL.get(task, 272)
    decoders = ["viterbi", "marginal"] if sweep else [decoder]

    chunk_size = (total + chunks - 1) // chunks
    print(f"FA proactive | task={task} decoders={decoders} seed={seed} "
          f"T={steps} gen_length={gen_length} total={total} chunks={chunks}")
    if task != "json":
        print(f"  non-regular grammar: bounded to depth {max_depth}; "
              f"coverage is reported per run and must accompany validity")

    handles = []
    for dec in decoders:
        if instance_ids:
            handles.append((dec, 0, run_chunk.spawn(
                task, dec, seed, len(instance_ids.split(",")), 0, steps,
                gen_length, max_depth, max_edges, temperature, instance_ids, tag)))
            continue
        for i in range(chunks):
            offset = i * chunk_size
            limit = min(chunk_size, total - offset)
            if limit <= 0:
                break
            print(f"  spawn {dec} chunk {i}: offset={offset} limit={limit}")
            handles.append((dec, offset, run_chunk.spawn(
                task, dec, seed, limit, offset, steps, gen_length,
                max_depth, max_edges, temperature, "", tag)))

    if nowait:
        print(f"\nspawned {len(handles)} container(s); not waiting.")
        print("  watch:   https://modal.com/apps  (or `modal app logs <app-id>`)")
        print(f"  fetch:   modal volume get dgrammar-results "
              f"'fa_*_{task}_s{seed}_t{steps}*.jsonl' results/")
    else:
        for dec, offset, handle in handles:
            print(f"\n{'=' * 60}\n=== {dec} offset={offset} ===\n{'=' * 60}")
            print(handle.get())

    print("\nMerge shards and compare with:")
    pats = " \\\n        ".join(
        f"--method fa_{d} 'results/fa_{d}_{task}_s{seed}_t{steps}*.jsonl'"
        for d in decoders
    )
    print(f"    python bench/measure_degeneracy.py \\\n        {pats} --common")

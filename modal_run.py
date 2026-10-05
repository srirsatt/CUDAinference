"""
modal_run.py - run any command from this repo on a Modal GPU.

Usage (from the repo root on your laptop):
    modal run modal_run.py --cmd "nvidia-smi"
    modal run modal_run.py --cmd "make ARCH=sm_75 copy && ./bin/copy"        # T4 (default)
    GPU=L4 modal run modal_run.py --cmd "make ARCH=sm_89 copy && ./bin/copy"
    GPU=H100 modal run modal_run.py --cmd "make ARCH=sm_90a copy && ./bin/copy"

Download files the run produces (e.g. a profiler report):
    modal run modal_run.py --cmd "make ARCH=sm_75 copy && ncu --set full -o copy_report ./bin/copy" \
        --get copy_report.ncu-rep

Your local repo is synced into the container on every run, so just edit
locally and re-run. Nothing persists between runs except what you --get.
"""
import os
import pathlib
import modal

# Which GPU to use. Set with the GPU environment variable; defaults to T4
# (the cheapest). Common choices: T4, L4, A10, A100-40GB, A100-80GB, H100, B200
GPU = os.environ.get("GPU", "T4")

# CUDA development image: includes nvcc and the CUDA headers/libraries.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("make")
    # Mount the local repo at /root/repo. Skip build outputs, git history,
    # and profiler reports so uploads stay small.
    .add_local_dir(
        ".",
        remote_path="/root/repo",
        ignore=["bin", ".git", "*.ncu-rep", "*.nsys-rep", "__pycache__"],
    )
)

app = modal.App("llm-inference-kernels", image=image)


@app.function(gpu=GPU, timeout=20 * 60)
def run(cmd: str, get: list[str]) -> dict[str, bytes]:
    import subprocess

    print(f"=== GPU: {GPU} | running: {cmd}\n", flush=True)
    result = subprocess.run(cmd, shell=True, cwd="/root/repo")

    # Collect any requested output files before the container disappears.
    files = {}
    for rel in get:
        path = pathlib.Path("/root/repo") / rel
        if path.exists():
            files[rel] = path.read_bytes()
        else:
            print(f"warning: requested file not found: {rel}")

    if result.returncode != 0:
        print(f"\n=== command exited with code {result.returncode}")
    return files


@app.local_entrypoint()
def main(cmd: str = "nvidia-smi", get: str = ""):
    wanted = [p.strip() for p in get.split(",") if p.strip()]
    files = run.remote(cmd, wanted)

    # Save downloaded files into ./modal_out/ on your laptop.
    if files:
        out = pathlib.Path("modal_out")
        out.mkdir(exist_ok=True)
        for rel, data in files.items():
            dest = out / pathlib.Path(rel).name
            dest.write_bytes(data)
            print(f"saved {dest}")
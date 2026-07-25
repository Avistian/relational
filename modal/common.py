"""Shared Modal harness for the relational curriculum (NOTES standard #20, rung 5).

Why this exists: the authoring box is a 12-core CPU with no GPU. When a lesson needs a
*verified* number that the laptop cannot produce — a trained transformer, a GNN over
RelBench, or simply a CV grid that would take hours — the agent escalates to Modal, which
runs unattended from a CLI token and bills per second ($30/month of free credits on the
Starter plan ≈ 50 T4-hours).

Pattern copied from ~/Projects/curie-llm/modal/common.py: one App, one Volume for
artifacts, images pinned here, thin per-lesson job files that import from this module.

Layout:
    modal/common.py          <- this file (App, images, volume, smoke test)
    modal/l0NN_<thing>.py    <- one job file per lesson that needs cloud compute

Usage:
    modal run modal/common.py                 # smoke-test the CPU image (~free)
    modal run modal/common.py --gpu           # also attach a T4 and check torch.cuda (~$0.01)
    modal run modal/l042_resnet.py            # a real job (example name)
    modal run --detach modal/<job>.py         # long job: survives the local CLI exiting

Artifacts land on the `relational-artifacts` volume; pull them with
    modal volume get relational-artifacts <remote-path> <local-path>
and commit the resulting numbers into the lesson + learning record like any other
measurement (state the hardware — a cloud number is not a laptop number).
"""

from __future__ import annotations

import modal

app = modal.App("relational-labs")

# ── Volume: everything a job produces that the lesson needs to quote ──────────
artifacts = modal.Volume.from_name("relational-artifacts", create_if_missing=True)
ARTIFACTS_PATH = "/artifacts"
volumes = {ARTIFACTS_PATH: artifacts}

# Kept in sync with ../requirements-labs.txt (minus jupyter/ipykernel, which a batch
# job does not need). torch is deliberately NOT here: it is the one heavy dependency,
# so it lives in `image_gpu` only and the CPU image stays small and fast to build.
LAB_DEPS = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "scikit-learn>=1.5",
    "imbalanced-learn>=0.12",
    "xgboost>=2.0",
    "lightgbm>=4.0",
    "catboost>=1.2",
    "pyarrow>=15.0",
]

_PY = "3.12"  # matches the local .venv, so results are comparable to laptop runs

image_cpu = (
    modal.Image.debian_slim(python_version=_PY)
    .pip_install(*LAB_DEPS)
    .add_local_dir("labs/relkit", remote_path="/root/relkit", copy=True)
)

# Default (CUDA) torch wheels — the whole point of the GPU image.
image_gpu = (
    modal.Image.debian_slim(python_version=_PY)
    .pip_install(*LAB_DEPS, "torch>=2.2")
    .add_local_dir("labs/relkit", remote_path="/root/relkit", copy=True)
)


@app.function(image=image_cpu, volumes=volumes, cpu=8, memory=16384, timeout=60 * 60)
def cpu_smoke() -> dict:
    """Prove the CPU image builds, imports the lab stack, and can write an artifact."""
    import os
    import platform

    import lightgbm
    import numpy as np
    import sklearn

    # Same thread-contention lesson as L031/L036: pin threads in cloud jobs too.
    n_cores = os.cpu_count()
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(2000, 20)), rng.integers(0, 2, 2000)
    m = lightgbm.LGBMClassifier(n_estimators=50, n_jobs=4, verbose=-1).fit(X, y)

    with open(f"{ARTIFACTS_PATH}/smoke.txt", "w") as fh:
        fh.write("cpu image ok\n")
    artifacts.commit()

    return {
        "python": platform.python_version(),
        "cores": n_cores,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "fitted_trees": m.n_estimators_,
    }


@app.function(image=image_gpu, gpu="T4", volumes=volumes, timeout=60 * 60)
def gpu_smoke() -> dict:
    """Prove a GPU is actually attached and torch sees it before a lesson depends on it."""
    import torch

    # Return only plain stdlib types: the value is pickled and deserialised *locally*,
    # and the local venv has no torch. `torch.__version__` is a TorchVersion instance,
    # not a str, so returning it raw fails with DeserializationError.
    ok = torch.cuda.is_available()
    out = {"torch": str(torch.__version__), "cuda_available": ok}
    if ok:
        out["device"] = str(torch.cuda.get_device_name(0))
        a = torch.randn(2048, 2048, device="cuda")
        out["matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
    return out


@app.local_entrypoint()
def main(gpu: bool = False):
    """`modal run modal/common.py` (CPU) or `... --gpu` to also check a T4."""
    print("CPU image:", cpu_smoke.remote())
    if gpu:
        print("GPU image:", gpu_smoke.remote())

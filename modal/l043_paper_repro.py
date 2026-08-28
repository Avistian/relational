"""L043 paper-results scale-up (NOTES standard #25).

Unattended GPU training of the from-scratch TabNet against the paper's Adult number
and a larger Syn4 mask-reading run.

    ~/.local/bin/modal run --detach modal/l043_paper_repro.py --preset closer

Presets: smoke (seconds) · closer (T4, ~15–30 min) · paper (hours).
The same loop is inlined in Lab 043's post-EXIT cell so you can *read* it there;
this file exists so the job can run without a browser tab.
"""
from __future__ import annotations

import os
import sys

import modal

sys.path.insert(0, os.path.dirname(__file__))
from common import ARTIFACTS_PATH, app, artifacts, image_gpu, volumes  # noqa: E402

image = image_gpu.add_local_dir("labs", remote_path="/root/labs", copy=True)


@app.function(image=image, gpu="T4", volumes=volumes, timeout=4 * 3600, memory=32768)
def run(preset: str = "closer") -> dict:
    import json
    import os
    import sys

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PAPER_REPRO_OUT"] = f"{ARTIFACTS_PATH}/l043/paper_repro.json"
    os.makedirs(f"{ARTIFACTS_PATH}/l043", exist_ok=True)
    sys.path.insert(0, "/root/labs")
    os.chdir("/root/labs")
    import _paper_repro_l043 as harness

    out = harness.main(["--preset", preset])
    artifacts.commit()
    # stdlib types only — Modal unpickles the return value locally, without torch.
    return {
        "preset": preset,
        "hardware": str(out.get("hardware")),
        "wrote": os.environ["PAPER_REPRO_OUT"],
    }


@app.local_entrypoint()
def main(preset: str = "closer"):
    print(run.remote(preset))

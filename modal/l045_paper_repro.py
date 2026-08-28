"""L045 paper-results scale-up (NOTES standard #25).

Unattended GPU training of from-scratch TabTransformer on *full* Adult
(contextual vs context-free vs CatBoost, plus RTD at 3% labels).

    ~/.local/bin/modal run --detach modal/l045_paper_repro.py --preset closer
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
    import os
    import sys

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PAPER_REPRO_OUT"] = f"{ARTIFACTS_PATH}/l045/paper_repro.json"
    os.makedirs(f"{ARTIFACTS_PATH}/l045", exist_ok=True)
    sys.path.insert(0, "/root/labs")
    os.chdir("/root/labs")
    import _paper_repro_l045 as harness

    out = harness.main(["--preset", preset])
    artifacts.commit()
    return {"preset": preset, "hardware": str(out.get("hardware")),
            "wrote": os.environ["PAPER_REPRO_OUT"]}


@app.local_entrypoint()
def main(preset: str = "closer"):
    print(run.remote(preset))

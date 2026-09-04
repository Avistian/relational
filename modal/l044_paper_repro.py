"""L044 paper-results scale-up (NOTES standard #25).

Unattended GPU training of from-scratch NODE vs CatBoost on a Higgs subsample
(or Adult fallback), compared to Popov et al. 2020 Table 1.

    ~/.local/bin/modal run --detach modal/l044_paper_repro.py --preset paper

Self-contained App (does not import `common`) so a detached run can import this
file inside the container without a sibling `common.py`.
"""
from __future__ import annotations

import modal

app = modal.App("relational-l044-paper-repro")
artifacts = modal.Volume.from_name("relational-artifacts", create_if_missing=True)
ARTIFACTS_PATH = "/artifacts"
volumes = {ARTIFACTS_PATH: artifacts}

# Kept in sync with modal/common.py + requirements-labs.txt (no jupyter).
_LAB_DEPS = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "scikit-learn>=1.5",
    "imbalanced-learn>=0.12",
    "xgboost>=2.0",
    "lightgbm>=4.0",
    "catboost>=1.2",
    "pyarrow>=15.0",
    "torch>=2.2",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_LAB_DEPS)
    .add_local_dir("labs", remote_path="/root/labs", copy=True)
)


@app.function(image=image, gpu="T4", volumes=volumes, timeout=12 * 3600, memory=32768)
def run(preset: str = "closer") -> dict:
    import os
    import sys

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PAPER_REPRO_OUT"] = f"{ARTIFACTS_PATH}/l044/paper_repro.json"
    os.makedirs(f"{ARTIFACTS_PATH}/l044", exist_ok=True)
    sys.path.insert(0, "/root/labs")
    os.chdir("/root/labs")
    import _paper_repro_l044 as harness

    out = harness.main(["--preset", preset])
    artifacts.commit()
    return {"preset": preset, "hardware": str(out.get("hardware")),
            "wrote": os.environ["PAPER_REPRO_OUT"]}


@app.local_entrypoint()
def main(preset: str = "closer"):
    print(run.remote(preset))

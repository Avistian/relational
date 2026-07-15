"""Canonical Google Colab bootstrap for lab notebooks — single source of truth.

Colab opens a lab as a lone `.ipynb` in a blank `/content` runtime; it does NOT
clone the repo. So `from relkit import ...` and the boosters (xgboost/lightgbm/
catboost) are missing. The bootstrap cell below fixes that on Colab and is a
no-op on a local venv or Binder (which already have the repo + deps).

Build scripts prepend these cells; `scripts/add_colab_bootstrap.py` injects them
into any already-generated notebook (idempotent via the MARKER).
"""
from __future__ import annotations

# Unique marker used to detect whether a notebook already has the bootstrap.
MARKER = "@colab-bootstrap"

BOOTSTRAP_MD = """### Running on Google Colab?

Colab opens only this single file, so the course package (`relkit`) and the lab
dependencies (xgboost, lightgbm, catboost, …) are **not** present by default. The cell
below fixes that: on Colab it shallow-clones the course repo, installs
`requirements-labs.txt`, and switches into `labs/` so `relkit` imports and the data cache
resolve. **On a local venv or Binder it does nothing — just run it and continue.**"""

BOOTSTRAP_CODE = """# @colab-bootstrap — PROVIDED. Makes the lab self-sufficient on Google Colab; a no-op elsewhere.
import os, sys

if "google.colab" in sys.modules:
    if not os.path.isdir("/content/relational"):
        !git clone --depth 1 https://github.com/Avistian/relational.git /content/relational
    %pip install -q -r /content/relational/requirements-labs.txt
    os.chdir("/content/relational/labs")
    print("Colab ready — working dir:", os.getcwd())
else:
    print("Not on Colab — using the local environment as-is.")"""


def _cell(cell_type: str, source: str) -> dict:
    cell: dict = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def bootstrap_cells(*, split_source: bool = False) -> list[dict]:
    """Return the [markdown, code] bootstrap cells.

    split_source=True stores `source` as a list of lines (nbformat-on-disk style,
    matching the build scripts); otherwise it stays a single string.
    """
    md = _cell("markdown", BOOTSTRAP_MD)
    code = _cell("code", BOOTSTRAP_CODE)
    if split_source:
        for c in (md, code):
            c["source"] = c["source"].splitlines(keepends=True)
    return [md, code]

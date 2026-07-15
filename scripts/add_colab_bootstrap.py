"""Inject the Colab bootstrap cell into lab notebooks (idempotent).

Colab loads a lab as a single file without the repo, so `relkit` and the boosters
are missing. This adds the canonical bootstrap (labs/_colab.py) as the first code
cell of every committed lab notebook, right after its title cell. Re-running is
safe: notebooks that already carry the @colab-bootstrap marker are skipped.

Usage (from repo root):
    .venv/bin/python scripts/add_colab_bootstrap.py            # all labs + template
    .venv/bin/python scripts/add_colab_bootstrap.py labs/0024-grinsztajn-benchmark.ipynb
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "labs"))

from _colab import MARKER, bootstrap_cells  # noqa: E402


def _has_bootstrap(nb: dict) -> bool:
    for c in nb.get("cells", []):
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if MARKER in src:
            return True
    return False


def _insert_index(nb: dict) -> int:
    """Insert after a leading title markdown cell if present, else at the top."""
    cells = nb.get("cells", [])
    if cells and cells[0].get("cell_type") == "markdown":
        return 1
    return 0


def process(path: str) -> str:
    with open(path) as f:
        nb = json.load(f)
    if _has_bootstrap(nb):
        return "skip (already has bootstrap)"
    at = _insert_index(nb)
    nb["cells"][at:at] = bootstrap_cells(split_source=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    return f"added at cell {at}"


def main(argv: list[str]) -> None:
    if argv:
        targets = argv
    else:
        targets = sorted(glob.glob(os.path.join(ROOT, "labs", "[0-9][0-9][0-9][0-9]-*.ipynb")))
        template = os.path.join(ROOT, "labs", "LAB-TEMPLATE.ipynb")
        if os.path.exists(template):
            targets.append(template)
    for path in targets:
        rel = os.path.relpath(path, ROOT)
        print(f"{rel:55} {process(path)}")


if __name__ == "__main__":
    main(sys.argv[1:])

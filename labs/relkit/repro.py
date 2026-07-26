"""Reproduction plumbing — the boilerplate half of Lab 037.

Deliberately partial. This module holds only the parts of a run manifest that are
pure environment inspection and carry no pedagogical content: where the repository
is, what is installed, what machine this is. The interesting half — content
hashing, output fingerprinting, the perturbation probe, the named metric
estimators, the noise floor and the reproduction gate — is what the student writes
in `labs/0037-document-a-baseline-package.ipynb`, and what they should append here
afterwards so later labs can import it.

See lessons/0037-document-a-baseline-package.html.
"""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata
from pathlib import Path

__all__ = ["git_state", "env_versions", "host_info"]

TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost", "torch",
)


def git_state(repo: str | Path = ".") -> dict:
    """Commit and cleanliness of `repo`, or an explicit marker when it is not a checkout.

    A manifest that records a commit but not whether the tree was dirty is worse
    than one that records neither, because it looks trustworthy.
    """
    repo = Path(repo)

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None, "note": "not a git checkout"}
    status = _git("status", "--porcelain")
    return {"sha": sha[:12], "dirty": bool(status), "branch": _git("rev-parse", "--abbrev-ref", "HEAD")}


def env_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict:
    """Resolved versions of the packages that can move a number, plus the interpreter.

    Records what is *installed*, not what a requirements file asks for: the whole
    point of L037's lightgbm 4.5.0 finding is that those are different questions.
    """
    out = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def host_info() -> dict:
    """Machine identity. Thread-count effects and BLAS differences live here."""
    import os

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }

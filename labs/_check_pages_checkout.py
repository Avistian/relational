"""Run the actual Pages build using only files in the Git index.

Stage intended publication files first. This catches ignored local artifacts that
make workspace-only staging tests pass while GitHub's clean checkout fails.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check():
    with tempfile.TemporaryDirectory(prefix="rdl-pages-checkout-") as tmp:
        root = Path(tmp)
        # Archived upstream source can contain LFS pointers whose objects belong
        # to the upstream repository. Publish their pinned bytes, just as the
        # default actions/checkout does, without fetching model checkpoints.
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(["git", "checkout-index", "--all", "--prefix=" + tmp + "/"], cwd=ROOT, env=env, check=True)
        workflow = (root / ".github/workflows/pages.yml").read_text()
        block = workflow.split("      - name: Build site\n        run: |\n", 1)[1].split("\n      - name:", 1)[0]
        script = "set -eu\n" + "\n".join(line[10:] for line in block.splitlines() if line.startswith("          "))
        result = subprocess.run(["bash", "-c", script], cwd=root, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError("Clean Git checkout Pages build failed:\n" + result.stderr)
        assert (root / "public/index.html").is_file()
        print("PASS: complete Pages build from Git index (no ignored workspace files)")

if __name__ == "__main__":
    check()

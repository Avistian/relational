"""Check normalized package presence only; never inspect installed versions."""

import argparse
from pathlib import Path
import re


# Existing direct requirements absent from the historical lock; refresh lock: Piotr.
# Keep explicit: new gaps must fail, rather than being learned from current files.
BASELINE_GAPS = frozenset({
    "dill",
    "duckdb",
    "einops",
    "entmax",
    "fasttext-wheel",
    "h5py",
    "ml-dtypes",
    "pymetis",
    "pytorch-frame",
    "pytorch-tabnet",
    "rdkit",
    "relbench",
    "rtdl-num-embeddings",
    "rtdl-revisiting-models",
    "skrub",
    "torch-geometric",
    "torcheval",
})


def package_names(path):
    """Read leading package names; suffix constraints/markers are not validated.

    These files use named requirements, not pip options or recursive includes.
    Reject unsupported lines so a new include cannot silently evade coverage.
    """
    names = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        requirement = line.split("#", 1)[0].strip()
        if not requirement:
            continue
        match = re.match(
            r"([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)(?=$|[\s\[<>=!~;@])",
            requirement,
        )
        if match is None:
            raise ValueError(f"{path}:{number}: expected a named requirement")
        names.add(re.sub(r"[-_.]+", "-", match[1]).lower())
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path)
    parser.add_argument("lock", type=Path)
    args = parser.parse_args()
    try:
        required = package_names(args.requirements)
        locked = package_names(args.lock)
    except (OSError, ValueError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    missing = required - locked
    known = missing & BASELINE_GAPS
    unexpected = missing - BASELINE_GAPS
    print(f"requirements={len(required)} locked={len(locked)} "
          f"covered={len(required & locked)} baseline_gaps={len(known)}")
    print("Known gaps only; refresh lock: Piotr")
    for name in sorted(known):
        print(f"KNOWN GAP: {name}")
    for name in sorted(unexpected):
        print(f"UNEXPECTED GAP: {name}")
    if unexpected:
        print(f"FAIL: {len(unexpected)} new unlocked package(s)")
        return 1
    print("PASS: normalized package presence (versions not checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

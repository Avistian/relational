"""Verify relmini: pinned data hashes, no-leakage, double-run determinism, JSON receipt.
Local reproducibility evidence only; no network, no downloads, cache read-only."""
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402

_C = baseline.CACHE
_P = {  # sha256 pinned from the lead env profile (config/verify-projects.json, project `relational`)
    "db/drivers.parquet": "ba3461b83a1a467a856bde4059519140a5a31952ad24a5e944ea9dd5c5faceae",
    "db/races.parquet": "88bb260f03c9ec5c7b030969d7284531d5f8e8592d73174686ad01cdfc8f87ea",
    "db/results.parquet": "b8b0e39b27f318a7bd461f21bb0ba9ef44909ed569d287e0c6b2969c6bd2f598",
    "tasks/driver-dnf/train.parquet": "994e9bdd0cf8f1930ac6ac77a5aa417d829f0472f93544a847cd616df99f6604",
    "tasks/driver-dnf/val.parquet": "f8edfda92a493c64fa9bb2da2a1ea54346b5e33c232e6617c1412da613333242",
    "tasks/driver-dnf/test.parquet": "6598427cdb1afb5308aa6a1e88593126b539a98a79404c953181ccb25175b20e",
}
PINS = {k: {"path": str(_C / k), "sha256": v} for k, v in _P.items()}
DEFAULT_RECEIPT = Path(__file__).resolve().parent / "receipt.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hashes(pins):
    """Return a list of problem strings (empty == all pinned hashes match)."""
    problems = []
    for name, pin in pins.items():
        p = Path(pin["path"])
        if not p.is_file():
            problems.append(f"{name}: missing {p}")
        elif sha256_file(p) != pin["sha256"]:
            problems.append(f"{name}: sha256 mismatch")
    return problems


def run(pins=PINS, receipt=DEFAULT_RECEIPT):
    problems = check_hashes(pins)
    if problems:
        print("FAIL hash check:\n  " + "\n  ".join(problems))
        return 1
    # no-leakage is asserted inside baseline.fit_predict for train/val/test feature builds
    a, b = baseline.run(), baseline.run()
    if a != b:
        print("FAIL: double run metrics differ", a, b)
        return 1
    out = {"status": "PASS", "hashes_checked": sorted(pins), "leakage_check": "feature_cutoff < task date asserted",
           "metrics": a, "double_run_identical": True,
           "note": "LOCAL reproducibility evidence only; not RelBench-comparable"}
    Path(receipt).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("PASS", json.dumps(a, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(run())

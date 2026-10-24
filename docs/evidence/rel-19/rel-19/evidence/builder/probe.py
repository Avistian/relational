"""Builder sensitivity probes; these are not TDD RED or independent review.

Run from repository root with the existing .venv and -B. No cache writes.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
from tests.relmini import test_real_leakage as audit

name = "test_cached_joins_match_each_labels_strict_history"
if sys.argv[1] == "missing":
    with tempfile.TemporaryDirectory() as folder:
        with patch.object(audit.baseline, "CACHE", Path(folder)):
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.TestSuite([audit.RealLeakage(name)]))
    assert result.testsRun == 1 and len(result.skipped) == 1
    assert result.skipped[0][1].startswith("NOT_RUN:")
    print("MISSING_CACHE=NOT_RUN (skip is not audit completion)")
elif sys.argv[1] == "mutation":
    original = audit.baseline.build_features

    def corrupt(*args, **kwargs):
        frame = original(*args, **kwargs)
        # Keep the cutoff metadata legal while corrupting a computed feature.
        frame.loc[0, "f_n_prior"] += 1
        return frame

    with patch.object(audit.baseline, "build_features", corrupt):
        result = unittest.TextTestRunner(verbosity=2, failfast=True).run(
            unittest.TestSuite([audit.RealLeakage(name)]))
    assert len(result.failures) == 1 and not result.errors and not result.skipped
    print("MUTATION_DETECTED=YES (builder sensitivity only; not TDD RED)")
else:
    raise ValueError("expected missing or mutation")

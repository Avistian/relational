"""Verifier evidence only; run once from the reviewed checkout with python -B."""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from tests.relmini import test_real_leakage as audit

b = audit.baseline
date = pd.Timestamp("2000-01-02")
task = pd.DataFrame({"driverId": [1], "date": [date]})
drivers = pd.DataFrame({"driverId": [1], "dob": [pd.Timestamp("1970-01-01")]})
results = pd.DataFrame({"driverId": [1, 1, 1],
                        "date": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                        "grid": [2., 999., 999.], "position": [1., None, None],
                        "points": [3., 999., 999.]})
frame = b.build_features(task, results, drivers)
assert frame.loc[0, "f_n_prior"] == 1
assert frame.loc[0, "f_mean_grid"] == 2
assert frame.loc[0, "f_mean_points"] == 3
assert frame.loc[0, "feature_cutoff"] == pd.Timestamp("2000-01-01")
b.assert_no_leakage(frame)
print("STRICT_BOUNDARY_OK same-day/future result rows excluded by real build_features")

for delta in (0, 1):
    corrupt = frame.copy()
    corrupt["feature_cutoff"] = date + pd.Timedelta(days=delta)
    try:
        b.assert_no_leakage(corrupt)
    except AssertionError as exc:
        assert "leakage: 1 rows" in str(exc)
    else:
        raise AssertionError("guard accepted forbidden cutoff")
print("GUARD_REJECTION_OK equal/future timestamps")

original = b.build_features
def corrupt_aggregate(*args, **kwargs):
    frame = original(*args, **kwargs)
    b.assert_no_leakage(frame)
    frame.loc[0, "f_mean_points"] += 10
    return frame

stream = io.StringIO()
with patch.object(b, "build_features", corrupt_aggregate):
    outcome = unittest.TextTestRunner(stream=stream, failfast=True, verbosity=2).run(
        unittest.TestSuite([audit.RealLeakage("test_cached_joins_match_each_labels_strict_history")]))
print(stream.getvalue(), file=sys.stderr, end="")
assert len(outcome.failures) == 1 and not outcome.errors and not outcome.skipped
assert "test_real_leakage.py\", line 56" in outcome.failures[0][1]
print("INDEPENDENT_ORACLE_SENSITIVITY_OK legal cutoff with corrupted mean_points rejected; not TDD RED")

with patch.object(b, "CACHE", Path("/nonexistent/rel19-verifier-cache")), \
     patch.object(b, "load_real", side_effect=AssertionError("must not load absent cache")):
    outcome = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([audit.RealLeakage("test_cached_joins_match_each_labels_strict_history")]))
assert len(outcome.skipped) == 1 and not outcome.errors and not outcome.failures
assert outcome.skipped[0][1].startswith("NOT_RUN:")
print("MISSING_CACHE_NO_READ_OK skip explicitly NOT_RUN; not audit completion")

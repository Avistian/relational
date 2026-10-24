"""Cached rel-f1 cutoff audit. No training, downloads, or cache writes.

Run directly: .venv/bin/python -B -m unittest tests.relmini.test_real_leakage -v
Missing required cache is a NOT_RUN skip, never evidence of a completed audit.
The oracle below filters source rows independently of the cumulative/asof join.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs" / "relmini"))
import baseline  # noqa: E402


class RealLeakage(unittest.TestCase):
    def test_cached_joins_match_each_labels_strict_history(self):
        required = ["db/results.parquet", "db/drivers.parquet"] + [
            f"tasks/driver-dnf/{split}.parquet" for split in ("train", "val", "test")
        ]
        missing = [name for name in required if not (baseline.CACHE / name).is_file()]
        if missing:
            self.skipTest("NOT_RUN: missing local rel-f1 cache: " + ", ".join(missing))

        for split in ("train", "val", "test"):
            task, results, drivers = baseline.load_real(split)
            self.assertGreater(len(task), 0, f"{split}: empty labels")
            feats = baseline.build_features(task, results, drivers)
            self.assertEqual(len(feats), len(task))
            pd.testing.assert_frame_equal(
                feats[["driverId", "date"]], task[["driverId", "date"]].reset_index(drop=True),
                check_dtype=False,
            )
            baseline.assert_no_leakage(feats)
            histories = {key: group for key, group in results.groupby("driverId")}
            birthdays = drivers.set_index("driverId")["dob"]
            with_history = 0
            for row, label in enumerate(task.itertuples(index=False)):
                with self.subTest(split=split, row=row):
                    source = histories.get(label.driverId, results.iloc[:0])
                    prior = source[source["date"] < label.date]
                    actual = feats.iloc[row]
                    n = len(prior)
                    self.assertEqual(actual.f_n_prior, n)
                    self.assertEqual(actual.f_has_history, float(n > 0))
                    if n:
                        with_history += 1
                        self.assertEqual(actual.feature_cutoff, prior["date"].max())
                        self.assertLess(actual.feature_cutoff, label.date)
                    else:
                        self.assertTrue(pd.isna(actual.feature_cutoff))
                    self.assertAlmostEqual(actual.f_dnf_rate, prior["position"].isna().mean() if n else 0.5)
                    self.assertAlmostEqual(actual.f_mean_grid, prior["grid"].sum() / n if n else 10.0)
                    self.assertAlmostEqual(actual.f_mean_points, prior["points"].sum() / n if n else 0.0)
                    dob = birthdays.get(label.driverId, pd.NaT)
                    age = (label.date - dob).days / 365.25 if pd.notna(dob) else 30.0
                    self.assertAlmostEqual(actual.f_age_years, age)
            self.assertGreater(with_history, 0, f"{split}: no historical joins exercised")
            print(f"CACHED_CUTOFF_OK split={split} labels={len(task)} with_history={with_history}")

    def test_future_dated_feature_is_rejected(self):
        # Explicitly synthetic: checks the existing guard, not a real-data claim.
        feats = pd.DataFrame({"date": [pd.Timestamp("2000-01-02")],
                              "feature_cutoff": [pd.Timestamp("2000-01-01")]})
        baseline.assert_no_leakage(feats)
        for cutoff in ("2000-01-02", "2000-01-03"):
            with self.subTest(cutoff=cutoff):
                corrupted = feats.copy()
                corrupted["feature_cutoff"] = pd.Timestamp(cutoff)
                with self.assertRaisesRegex(AssertionError, "leakage: 1 rows"):
                    baseline.assert_no_leakage(corrupted)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for labs/relmini. Fixtures here are SYNTHETIC; the real-data test uses the local rel-f1 cache."""
import json, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs" / "relmini"))  # not a package: tests/relmini would shadow it
import baseline, verify  # noqa: E402

CACHE = Path.home() / ".cache" / "relbench" / "rel-f1"


def synthetic(n_drivers=40, n_races=60, seed=1):
    """SYNTHETIC drivers/results/tasks. Outcome depends on driver skill, so a model can learn."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n_races, freq="14D")
    skill = rng.uniform(0, 1, n_drivers)
    drivers = pd.DataFrame({"driverId": range(n_drivers),
                            "dob": pd.Timestamp("1970-01-01") + pd.to_timedelta(rng.integers(0, 3000, n_drivers), "D")})
    rows = []
    for r, d in enumerate(dates):
        for dr in range(n_drivers):
            dnf = rng.uniform() > skill[dr]
            rows.append(dict(raceId=r, driverId=dr, date=d, grid=int(rng.integers(1, 20)),
                             position=np.nan if dnf else float(rng.integers(1, 20)), points=0.0 if dnf else 1.0,
                             did_not_finish=int(dnf)))
    res = pd.DataFrame(rows)
    task = res[["date", "driverId", "did_not_finish"]].copy()
    return drivers, res.drop(columns="did_not_finish"), task


class LeakageGuard(unittest.TestCase):
    def test_cutoff_strictly_before_task_timestamp(self):
        drivers, res, task = synthetic()
        feats = baseline.build_features(task, res, drivers)
        has = feats["feature_cutoff"].notna()
        self.assertTrue(has.any())
        self.assertTrue((feats.loc[has, "feature_cutoff"] < feats.loc[has, "date"]).all())
        baseline.assert_no_leakage(feats)

    def test_features_ignore_same_day_and_future_results(self):
        drivers, res, task = synthetic()
        base = baseline.build_features(task, res, drivers)
        # corrupt every result on/after the 30th race date: earlier-row features must not change
        cut = sorted(res.date.unique())[30]
        res2 = res.copy()
        m = res2.date >= cut
        res2.loc[m, ["grid", "points"]] = 999.0
        res2.loc[m, "position"] = np.nan
        alt = baseline.build_features(task, res2, drivers)
        early = base.date <= cut  # rows at `cut` may only see results before `cut`
        cols = [c for c in base.columns if c.startswith("f_")]
        pd.testing.assert_frame_equal(base.loc[early, cols], alt.loc[early, cols])

    def test_guard_raises_on_leaky_frame(self):
        drivers, res, task = synthetic(n_races=5)
        feats = baseline.build_features(task, res, drivers)
        feats.loc[feats.index[-1], "feature_cutoff"] = feats.loc[feats.index[-1], "date"]
        with self.assertRaises(AssertionError):
            baseline.assert_no_leakage(feats)


class Determinism(unittest.TestCase):
    def test_same_seed_identical_predictions(self):
        drivers, res, task = synthetic()
        tr, va = task[task.date < "2001-06-01"], task[task.date >= "2001-06-01"]
        a = baseline.fit_predict(tr, va, res, drivers, seed=0)
        b = baseline.fit_predict(tr, va, res, drivers, seed=0)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), len(va))
        self.assertTrue(((a >= 0) & (a <= 1)).all())

    def test_model_learns_on_synthetic(self):
        drivers, res, task = synthetic()
        tr, va = task[task.date < "2001-06-01"], task[task.date >= "2001-06-01"]
        p = baseline.fit_predict(tr, va, res, drivers, seed=0)
        self.assertGreater(baseline.auroc(va.did_not_finish.to_numpy(), p), 0.6)


class HashCheck(unittest.TestCase):
    def test_pass_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "x.parquet"
            f.write_bytes(b"hello")
            good = {"x": {"path": str(f), "sha256": verify.sha256_file(f)}}
            self.assertEqual(verify.check_hashes(good), [])
            bad = {"x": {"path": str(f), "sha256": "0" * 64}}
            self.assertEqual(len(verify.check_hashes(bad)), 1)
            missing = {"y": {"path": str(Path(t) / "nope"), "sha256": "0" * 64}}
            self.assertEqual(len(verify.check_hashes(missing)), 1)

    def test_verify_main_fails_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "x"
            f.write_bytes(b"a")
            rc = verify.run({"x": {"path": str(f), "sha256": "1" * 64}}, receipt=Path(t) / "r.json")
            self.assertNotEqual(rc, 0)
            self.assertFalse((Path(t) / "r.json").exists())


@unittest.skipUnless((CACHE / "db" / "results.parquet").exists(),
                     "SKIPPED: local RelBench rel-f1 cache absent at ~/.cache/relbench/rel-f1 (no download attempted)")
class RealData(unittest.TestCase):
    def test_real_driver_dnf_no_leakage_and_pins(self):
        self.assertEqual(verify.check_hashes(verify.PINS), [])
        task, res, drivers = baseline.load_real("val")
        feats = baseline.build_features(task, res, drivers)
        baseline.assert_no_leakage(feats)
        self.assertEqual(len(feats), len(task))


if __name__ == "__main__":
    unittest.main()

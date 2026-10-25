"""Constructor ablation behavior. In-memory examples are SYNTHETIC only."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'labs/relmini'))
import baseline


class ConstructorFeatures(unittest.TestCase):
    def test_constructor_history_uses_latest_prior_team_and_strict_cutoff(self):
        task = pd.DataFrame({'driverId': [1, 1, 2], 'date': pd.to_datetime(['2000-01-03', '2000-01-02', '2000-01-01'])})
        drivers = pd.DataFrame({'driverId': [1, 2], 'dob': pd.to_datetime(['1970-01-01'] * 2)})
        results = pd.DataFrame({'driverId': [1, 1, 1], 'constructorId': [10, 20, 30],
                                'date': pd.to_datetime(['2000-01-01', '2000-01-02', '2000-01-03']),
                                'grid': [1., 2., 99.], 'position': [1., 2., 99.], 'points': [2., 3., 99.]})
        constructors = pd.DataFrame({'constructorId': [10, 20, 20, 20],
                                    'date': pd.to_datetime(['2000-01-01', '2000-01-01', '2000-01-03', '2000-01-04']),
                                    'points': [7., 11., 999., 999.]})
        f = baseline.build_features(task, results, drivers, constructor_results=constructors)
        self.assertEqual(f.f_constructor_mean_points.tolist(), [11., 7., 0.])
        self.assertEqual(f.f_constructor_n_prior.tolist(), [1., 1., 0.])
        self.assertEqual(f.f_constructor_has_history.tolist(), [1., 1., 0.])
        self.assertEqual(f.constructor_cutoff.iloc[0], pd.Timestamp('2000-01-01'))
        self.assertTrue(pd.isna(f.constructor_cutoff.iloc[2]))
        pd.testing.assert_frame_equal(f[['driverId', 'date']], task)
        baseline.assert_no_leakage(f)
        # A same-day driver transfer and same-day/future constructor scores cannot change features.
        changed = constructors.copy()
        changed.loc[changed.date >= pd.Timestamp('2000-01-03'), 'points'] = -500.
        alt = baseline.build_features(task, results, drivers, constructor_results=changed)
        pd.testing.assert_frame_equal(f, alt)


if __name__ == '__main__':
    unittest.main()


class ConstructorGuard(unittest.TestCase):
    def test_guard_rejects_constructor_and_assignment_at_or_after_label(self):
        date = pd.Timestamp('2000-01-03')
        for column in ('constructor_cutoff', 'constructor_assignment_cutoff'):
            for delta in (0, 1):
                f = pd.DataFrame({'date': [date], 'feature_cutoff': [date - pd.Timedelta(days=2)],
                                  column: [date + pd.Timedelta(days=delta)]})
                with self.subTest(column=column, delta=delta):
                    with self.assertRaisesRegex(AssertionError, 'leakage'):
                        baseline.assert_no_leakage(f)


class PairedAblation(unittest.TestCase):
    def test_same_model_without_constructor_features_reproduces_baseline(self):
        import numpy as np
        from tests.relmini.test_relmini import synthetic
        drivers, results, task = synthetic(n_drivers=12, n_races=20)
        results['constructorId'] = results.driverId % 3
        constructors = results.groupby(['constructorId', 'date'], as_index=False).agg(points=('points', 'sum'))
        train, ev = task[task.date < '2000-07-01'], task[task.date >= '2000-07-01']
        original = baseline.fit_predict(train, ev, results, drivers, seed=0)
        without = baseline.fit_predict(train, ev, results, drivers, seed=0,
                                       constructor_results=constructors, use_constructor=False)
        with_team = baseline.fit_predict(train, ev, results, drivers, seed=0,
                                         constructor_results=constructors, use_constructor=True)
        again = baseline.fit_predict(train, ev, results, drivers, seed=0,
                                     constructor_results=constructors, use_constructor=True)
        np.testing.assert_array_equal(original, without)
        np.testing.assert_array_equal(with_team, again)
        self.assertEqual(len(with_team), len(ev))
        self.assertTrue(np.isfinite(with_team).all())
        self.assertFalse(np.array_equal(with_team, without))
        with self.assertRaisesRegex(ValueError, 'constructor'):
            baseline.fit_predict(train, ev, results, drivers, use_constructor=True)


class AblationReceipt(unittest.TestCase):
    def test_missing_cache_records_not_run_without_results(self):
        import json
        import tempfile
        import ablation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = ablation.run(cache=root / 'missing', output=root / 'out', smoke=True)
            self.assertNotEqual(rc, 0)
            receipt = json.loads((root / 'out' / 'receipt.json').read_text())
            self.assertEqual(receipt['status'], 'NOT_RUN')
            self.assertNotIn('metrics', receipt)
            self.assertFalse((root / 'out' / 'predictions.csv').exists())


class AmbiguousTeams(unittest.TestCase):
    def test_conflicting_latest_team_is_unknown_not_an_arbitrary_choice(self):
        drivers = pd.DataFrame({'driverId': [1], 'dob': pd.to_datetime(['1970-01-01'])})
        task = pd.DataFrame({'driverId': [1], 'date': pd.to_datetime(['2000-01-03'])})
        results = pd.DataFrame({'driverId': [1, 1, 1], 'constructorId': [10, 10, 20],
                                'date': pd.to_datetime(['2000-01-01', '2000-01-02', '2000-01-02']),
                                'grid': [1., 1., 1.], 'position': [1., 1., 1.], 'points': [1., 1., 1.]})
        constructors = pd.DataFrame({'constructorId': [10, 20],
                                    'date': pd.to_datetime(['2000-01-01'] * 2), 'points': [7., 11.]})
        f = baseline.build_features(task, results, drivers, constructors)
        self.assertEqual(f.f_constructor_has_history.iloc[0], 0.)
        self.assertEqual(f.f_constructor_n_prior.iloc[0], 0.)
        self.assertTrue(pd.isna(f.constructor_cutoff.iloc[0]))
        # Reversing tied source rows must not choose a different team.
        alt = baseline.build_features(task, results.iloc[::-1], drivers, constructors)
        pd.testing.assert_frame_equal(f, alt)


class EvidencePreservation(unittest.TestCase):
    def test_existing_evidence_is_never_overwritten_by_another_run(self):
        import tempfile
        import ablation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'out'
            output.mkdir()
            receipt = output / 'receipt.json'
            receipt.write_text('historic evidence\n')
            with self.assertRaises(FileExistsError):
                ablation.run(cache=root / 'missing', output=output, smoke=True)
            self.assertEqual(receipt.read_text(), 'historic evidence\n')

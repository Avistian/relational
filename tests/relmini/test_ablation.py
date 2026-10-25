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

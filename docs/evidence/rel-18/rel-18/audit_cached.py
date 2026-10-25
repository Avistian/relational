"""Independent strict-filter oracle on every cached label; no fitting or cache writes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'labs' / 'relmini'))
import baseline

constructors = pd.read_parquet(baseline.CACHE / 'db/constructor_results.parquet')
constructor_groups = dict(tuple(constructors.groupby('constructorId')))
for split in ('train', 'val', 'test'):
    task, results, drivers = baseline.load_real(split)
    feats = baseline.build_features(task, results, drivers, constructors)
    original = baseline.build_features(task, results, drivers)
    pd.testing.assert_frame_equal(feats[baseline.FEATURES], original[baseline.FEATURES])
    baseline.assert_no_leakage(feats)
    driver_groups = dict(tuple(results.groupby('driverId')))
    history_count, ambiguous = 0, 0
    for i, label in enumerate(task.itertuples(index=False)):
        actual = feats.iloc[i]
        history = driver_groups.get(label.driverId, results.iloc[:0])
        history = history[history.date < label.date]
        prior = constructors.iloc[:0]
        if len(history):
            latest = history.date.max()
            assert actual.constructor_assignment_cutoff == latest
            teams = history.loc[history.date == latest, 'constructorId'].unique()
            if len(teams) == 1:
                prior = constructor_groups.get(teams[0], constructors.iloc[:0])
                prior = prior[prior.date < label.date]
            else:
                ambiguous += 1
        else:
            assert pd.isna(actual.constructor_assignment_cutoff)
        n = len(prior)
        assert actual.f_constructor_n_prior == n, (split, i)
        assert actual.f_constructor_has_history == float(n > 0), (split, i)
        np.testing.assert_allclose(actual.f_constructor_mean_points, prior.points.sum() / n if n else 0.)
        if n:
            history_count += 1
            assert actual.constructor_cutoff == prior.date.max()
        else:
            assert pd.isna(actual.constructor_cutoff)
    print(f'ACTUAL_CONSTRUCTOR_ORACLE_PASS split={split} rows={len(task)} with_history={history_count} ambiguous={ambiguous}')

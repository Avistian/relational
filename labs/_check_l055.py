"""Behavioral checks for the temporal evaluation contract (no network)."""
import numpy as np
from relkit.temporal import chronological_split, fit_preprocessor, apply_preprocessor, select_candidate, rank_change

t = np.array([4, 1, 3, 2, 3, 6, 5, 7, 8, 9])
s = chronological_split(t, 4, 7)
assert set(s['train']) == {1, 2, 3, 4}
assert set(s['val']) == {0, 5, 6}
assert set(s['test']) == {7, 8, 9}
assert max(t[s['train']]) < min(t[s['val']])
assert max(t[s['val']]) < min(t[s['test']])
x = np.array([[1., np.nan], [3., np.nan], [999., 9.]])
p = fit_preprocessor(x[:2])
assert np.allclose(apply_preprocessor(x[:2], p), [[-1, 0], [1, 0]])
assert np.isfinite(apply_preprocessor(x, p)).all()
assert np.allclose(p[1], [2, 0]), 'Held-out values must not affect training statistics'
assert select_candidate([.3, .2, .2]) == 1, 'First validation tie wins'
assert rank_change([.1, .2, .3], [.3, .2, .1]).tolist() == [2., 0., -2.]
print('PASS: timestamp ties, strict boundaries, train-only preprocessing, validation selection, rank direction')

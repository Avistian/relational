"""Behavioral checks for the checkpoint's experiment boundaries."""
import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).parent / 'relkit/checkpoint.py'
assert path.exists(), 'Checkpoint needs a training-only preprocessing and selection implementation.'
from relkit.checkpoint import prepare_numeric, select_trial, paired_summary, reference_parity

x = np.array([[0.], [2.], [100.], [200.]])
z, state = prepare_numeric(x, np.array([0, 1]))
assert np.allclose(z[:2, 0], [-1, 1]), 'Held-out rows changed the training scale.'
x[2:] *= 100
z2, state2 = prepare_numeric(x, np.array([0, 1]))
assert state == state2 and np.array_equal(z[:2], z2[:2])
assert select_trial([.8, .9, .9]) == 1, 'Choose the first validation maximum.'
result = paired_summary([.8, .82, .84], [.79, .81, .83])
assert np.isclose(result['mean'], .01) and result['sd'] < 1e-12
print(reference_parity())
print('PASS: held-out intervention, validation selection, paired uncertainty, reference parity')

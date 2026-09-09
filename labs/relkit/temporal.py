"""L055: visible split, preprocessing, selection and rank operations.

The experimental intervention is the evaluation protocol, not a new model.
"""
import numpy as np
from scipy.stats import rankdata


def chronological_split(time, validation_start, test_start):
    """Whole timestamp groups; half-open intervals. No label access."""
    time = np.asarray(time)
    if validation_start >= test_start or not np.isfinite(time).all():
        raise ValueError('Need finite times and increasing cutoffs')
    parts = dict(train=np.flatnonzero(time < validation_start),
                 val=np.flatnonzero((time >= validation_start) & (time < test_start)),
                 test=np.flatnonzero(time >= test_start))
    if any(len(v) == 0 for v in parts.values()):
        raise ValueError('Every partition needs rows')
    return parts


def fit_preprocessor(train):
    """Median imputation then mean/std scaling, fitted only on training rows."""
    train = np.where(np.isfinite(train), train, np.nan).astype(float)
    median = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in train.T])
    filled = np.where(np.isfinite(train), train, median)
    mean = filled.mean(0)
    scale = filled.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    return median, mean, scale


def apply_preprocessor(x, state):
    median, mean, scale = state
    return ((np.where(np.isfinite(x), x, median) - mean) / scale).astype('float32')


def select_candidate(validation_errors):
    """Lower-is-better validation metric; first minimum wins ties."""
    errors = np.asarray(validation_errors)
    if not np.isfinite(errors).all() or not errors.size:
        raise ValueError('Need nonempty finite validation errors')
    return int(np.argmin(errors))


def rank_change(random_errors, temporal_errors):
    """Positive means a worse rank on the temporal protocol; average ties."""
    return rankdata(temporal_errors) - rankdata(random_errors)

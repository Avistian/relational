"""L055 v2: same-pool split intervention; historical temporal.py remains measured."""
import numpy as np
from relkit.temporal import chronological_split, fit_preprocessor, apply_preprocessor, select_candidate, rank_change


def paired_random_split(temporal_ids, seed):
    """Shuffle one temporal row pool, retaining partition sizes; never inspect labels."""
    keys = ('train', 'val', 'test')
    parts = [np.asarray(temporal_ids[k], dtype=np.int64) for k in keys]
    if any(v.ndim != 1 or len(v) == 0 for v in parts):
        raise ValueError('Need three nonempty one-dimensional partitions')
    pool = np.concatenate(parts)
    if len(np.unique(pool)) != len(pool):
        raise ValueError('Row IDs must be disjoint and unique')
    shuffled = np.random.default_rng(seed).permutation(pool)
    ends = np.cumsum([len(v) for v in parts])[:-1]
    return {k: np.sort(v) for k, v in zip(keys, np.split(shuffled, ends))}

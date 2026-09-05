"""L047 data and evaluation protocol; no model implementation hidden here."""
from __future__ import annotations
import hashlib
import importlib.metadata
import platform
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, t
from sklearn.model_selection import train_test_split
from relkit.data import load_tier_a, SPECS


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def prepare(name, split_seed=5, cap=None, split='stratified'):
    X, y = load_tier_a(name)
    row_ids = np.arange(len(X))
    if cap and cap < len(X):
        row_ids, _ = train_test_split(row_ids, train_size=cap,
                                     random_state=17, stratify=y)
        row_ids.sort()
    X, y = X.iloc[row_ids].reset_index(drop=True), np.asarray(y.iloc[row_ids]).copy()
    if split == 'released':
        # Matches data_openml.py's RNG/assignment; proportions fluctuate.
        assignment = np.random.RandomState(split_seed).choice(3, size=len(X), p=[.65, .15, .20])
        tr, va, te = [np.flatnonzero(assignment == i) for i in range(3)]
    else:
        tr, te = train_test_split(np.arange(len(X)), test_size=.20,
                                 random_state=split_seed, stratify=y)
        tr, va = train_test_split(tr, test_size=.15/.80,
                                 random_state=split_seed, stratify=y[tr])
        tr, va, te = map(np.sort, (tr, va, te))
    cats = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X if c not in cats]
    mappings = [{v: i+1 for i, v in enumerate(sorted(X.iloc[tr][c].dropna().astype(str).unique()))}
                for c in cats]
    xc = np.column_stack([X[c].astype('string').map(m).fillna(0).to_numpy(dtype=np.int64)
                          for c, m in zip(cats, mappings)]) if cats else np.empty((len(X), 0), np.int64)
    raw_num = X[nums].to_numpy(dtype=np.float32)
    mean, std = np.nanmean(raw_num[tr], axis=0), np.nanstd(raw_num[tr], axis=0)
    mean = np.nan_to_num(mean)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.)
    xn = ((raw_num - mean) / std).astype(np.float32)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()
                               + y.tobytes()).hexdigest()
    meta = {'dataset': name, 'openml_id': SPECS[name]['openml_id'], 'rows': len(X),
            'data_sha256': data_hash, 'split_seed': split_seed, 'split_method': split,
            'subsample_seed': 17 if cap else None,
            'row_ids': row_ids.tolist(), 'train': tr.tolist(), 'valid': va.tolist(), 'test': te.tolist(),
            'split_hashes': {k: digest(v) for k, v in zip(('train','valid','test'), (tr,va,te))},
            'numeric_columns': nums, 'categorical_columns': cats,
            'train_numeric_mean': mean.tolist(), 'train_numeric_std': std.tolist(),
            'categorical_vocabularies': mappings,
            'evaluation_context': 'within-split, ascending row index, fixed batches; no labels in forward'}
    return dict(xn=xn, xc=xc, y=y, cards=[len(m)+1 for m in mappings],
                train=tr, valid=va, test=te, meta=meta)


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'versions': {p: importlib.metadata.version(p) for p in
                         ('torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost')}}


def summarize(per_dataset, models):
    means, summary = [], {}
    for name, rows in per_dataset.items():
        summary[name] = {}
        for model in models:
            scores = np.asarray([r[model] for r in rows])
            sd = float(scores.std(ddof=1)) if len(scores) > 1 else 0.
            half = float(t.ppf(.975, len(scores)-1) * sd / len(scores)**.5) if len(scores)>1 else None
            summary[name][model] = {'mean': float(scores.mean()), 'sample_std': sd,
                                    'seed_ci95_halfwidth': half, 'scores': scores.tolist()}
        means.append([summary[name][m]['mean'] for m in models])
    score = np.asarray(means)
    ranks = rankdata(-score, axis=1).mean(axis=0)
    result = {'per_dataset': summary, 'mean_ranks': dict(zip(models, ranks.tolist()))}
    if len(score) >= 3 and len(models) >= 3:
        f = friedmanchisquare(*score.T)
        k, n = len(models), len(score)
        cd = studentized_range.ppf(.95, k, np.inf) / np.sqrt(2) * np.sqrt(k*(k+1)/(6*n))
        result.update(friedman={'statistic': float(f.statistic), 'p': float(f.pvalue)}, nemenyi_cd=float(cd))
    result['uncertainty_note'] = 'Seed CIs conditional on fixed split; not population CIs. Tiny dataset count limits rank-test power.'
    return result

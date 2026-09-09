"""L056: visible, dataset-balanced audit of already measured benchmark errors.

No model fitting or official Elo-table reproduction is performed here.
"""
import numpy as np
import pandas as pd


def rank_errors(errors):
    """Smaller error earns rank 1; exact ties receive their average rank."""
    e = np.asarray(errors, dtype=float)
    if e.ndim != 1 or len(e) == 0 or not np.isfinite(e).all():
        raise ValueError('Expected a nonempty finite error vector')
    return np.array([1 + np.sum(e < x) + (np.sum(e == x) - 1) / 2 for x in e])


def aligned_errors(rows, arms):
    """Require identical dataset/fold coverage and metrics, without imputation."""
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('Arms must be nonempty and unique')
    x = rows.loc[rows.arm.isin(arms)].copy()
    if x.empty or x.duplicated(['dataset','fold','arm']).any():
        raise ValueError('Empty or duplicate evaluation keys')
    if x.imputed.isna().any() or x.imputed.any():
        raise ValueError('This audit requires observed, non-imputed results')
    if not np.isfinite(x.metric_error).all():
        raise ValueError('Nonfinite metric error')
    if x.groupby('dataset').metric.nunique().ne(1).any():
        raise ValueError('Metrics differ within a dataset')
    wide = x.pivot(index=['dataset','fold'],columns='arm',values='metric_error').reindex(columns=arms)
    if wide.isna().any().any():
        raise ValueError('Unequal evaluation coverage; do not silently drop failures')
    return wide.sort_index()


def macro_ranks(split_ranks):
    """Average over each dataset's outer splits; keep one row per dataset."""
    if split_ranks.empty or not np.isfinite(split_ranks.to_numpy()).all():
        raise ValueError('Expected finite split ranks')
    return split_ranks.groupby(level='dataset',sort=True).mean()


def bootstrap_gap(dataset_gaps, seed=56, n_boot=2000):
    """Mean and percentile 95% interval, resampling whole paired datasets."""
    x = np.asarray(dataset_gaps,dtype=float)
    if x.ndim != 1 or len(x)<2 or not np.isfinite(x).all() or n_boot<100:
        raise ValueError('Need at least two finite dataset gaps and 100 draws')
    rng = np.random.default_rng(seed)
    draws = rng.choice(x,size=(n_boot,len(x)),replace=True).mean(axis=1)
    lo,hi = np.quantile(draws,[.025,.975])
    return np.array([x.mean(),lo,hi])


def summarize(rows, arms, n_boot=2000):
    """Calls the four live learner operations; all regimes use the same arms."""
    result = {}
    for regime in ('default','tuned','tuned_ensemble'):
        frame = rows.loc[rows.method_subtype == regime]
        errors = aligned_errors(frame,arms)
        ranks = pd.DataFrame([rank_errors(row) for row in errors.to_numpy()],index=errors.index,columns=arms)
        per_dataset = macro_ranks(ranks)
        # Pairwise win probability: ties count as half; lower error wins.
        wins = {}
        for a in arms:
            for b in arms:
                if a == b: continue
                v = ((errors[a] < errors[b]).astype(float)+.5*(errors[a]==errors[b]))
                wins[a+' / '+b] = float(v.groupby(level='dataset').mean().mean())
        gap = per_dataset['TabM']-per_dataset['CatBoost']
        # Supplementary Friedman/Nemenyi: rank MEAN error within each dataset.
        # It answers a different question from averaging split-level ranks.
        from scipy.stats import friedmanchisquare, studentized_range, rankdata
        means = errors.groupby(level='dataset').mean()
        friedman = friedmanchisquare(*means.to_numpy().T)
        sd = errors.groupby(level='dataset').std().reset_index().astype(object)
        sd = sd.where(pd.notna(sd),None)
        cd = float(studentized_range.ppf(.95,len(arms),np.inf)/np.sqrt(2)*np.sqrt(len(arms)*(len(arms)+1)/(6*len(means))))
        result[regime] = dict(datasets=len(per_dataset),outer_splits=len(errors),
            mean_ranks=per_dataset.mean().to_dict(),naive_split_ranks=ranks.mean().to_dict(),
            dataset_ranks=per_dataset.reset_index().to_dict(orient='records'),
            dataset_errors=means.reset_index().to_dict(orient='records'),
            split_sd=sd.to_dict(orient='records'),
            win_rates=wins,tabm_minus_catboost_rank_gap=bootstrap_gap(gap,n_boot=n_boot).tolist(),
            friedman_p=float(friedman.pvalue),nemenyi_cd=cd,
            mean_error_ranks=dict(zip(arms,rankdata(means.to_numpy(),axis=1).mean(axis=0))))
    return result

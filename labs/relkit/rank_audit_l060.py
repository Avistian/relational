"""Conditional permutation calibration of dataset-block rank dispersion."""
import itertools
import numpy as np


def rank_permutation(ranks,draws=20000,seed=60):
    r=np.asarray(ranks,float)
    if r.ndim!=2 or not r.size or not np.isfinite(r).all():raise ValueError('Finite dataset by method ranks required')
    n,k=r.shape
    # Equivalent to Friedman ordering for fixed within-row tie patterns.
    center=(k+1)/2
    observed=float(np.sum((r.mean(0)-center)**2))
    exceed=0;total=0
    if n==3 and k<=5:
        # Global relabeling fixes the first row without changing dispersion.
        permutations=list(itertools.permutations(range(k)))
        for p in permutations:
            for q in permutations:
                score=np.sum(((r[0]+r[1,list(p)]+r[2,list(q)])/3-center)**2)
                exceed+=int(score>=observed-1e-12);total+=1
        p=exceed/total;mode='exact conditional method-label permutation; first row fixed'
        mc_se=None
    else:
        rng=np.random.default_rng(seed)
        for _ in range(draws):
            score=np.sum((np.stack([rng.permutation(row) for row in r]).mean(0)-center)**2)
            exceed+=int(score>=observed-1e-12)
        total=draws;p=(1+exceed)/(1+draws);mc_se=float(np.sqrt(p*(1-p)/(draws+1)))
        mode='Monte Carlo conditional method-label permutation, plus-one correction'
    return dict(p=float(p),mc_se=mc_se,draws=total,exceed=exceed,statistic=observed,seed=seed,mode=mode,
                assumption='Within each independent task block, method labels are exchangeable under the null; the convenience panel is not a random sample of enterprises')

"""Visible B03 mechanisms; no claim to reproduce TabPFN pretraining."""
import numpy as np

def posterior_predictive(prior, likelihood, query_prob):
    """Exact finite-prior Bayesian prediction, not a trained neural approximation."""
    p,l,q=[np.asarray(x,dtype=float) for x in (prior,likelihood,query_prob)]
    if p.ndim!=1 or not len(p) or p.shape!=l.shape or p.shape!=q.shape:
        raise ValueError('Three equal nonempty vectors required')
    if not all(np.isfinite(x).all() for x in (p,l,q)) or np.any(p<0) or np.any(l<0) or np.any((q<0)|(q>1)):
        raise ValueError('Finite nonnegative weights and probabilities required')
    if not np.isclose(p.sum(),1) or l.max()<=0:raise ValueError('Normalized prior and nonzero likelihood required')
    weights=p*(l/l.max())
    if weights.sum()<=0:raise ValueError('Support impossible under prior')
    return float((weights/weights.sum())@q)

def align_probabilities(probabilities, old_to_new):
    """If labels transform c -> pi[c], restore old column c from new pi[c]."""
    p=np.asarray(probabilities,dtype=float);pi=np.asarray(old_to_new)
    if p.ndim!=2 or not p.shape[0] or pi.ndim!=1 or pi.dtype.kind not in 'iu' or len(pi)!=p.shape[1]:
        raise ValueError('Probability matrix and integer class permutation required')
    if not np.array_equal(np.sort(pi),np.arange(len(pi))):raise ValueError('Not a bijection')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('Invalid probability rows')
    return p[:,pi].copy()

def compare_variants(measured, claimed):
    """A matched contract is necessary, not sufficient, for score reproduction."""
    keys=('version','variant','checkpoint','recipe','data','splits','budget')
    if any(not isinstance(x.get(k),str) or x[k] in ('','UNKNOWN','NOT_RUN') for x in (measured,claimed) for k in keys):
        return 'INCOMPLETE'
    return 'MATCHED_CONTRACT' if all(measured[k]==claimed[k] for k in keys) else 'INCOMPARABLE'

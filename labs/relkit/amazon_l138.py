"""E-commerce task contracts; NumPy-only learner implementation."""
import numpy as np

def review_target(review_days,cutoff,horizon=91):
    """Return (eligible, churn); days share an origin. Ineligible has no label."""
    days=np.asarray(review_days,dtype=float)
    if days.ndim!=1 or not np.isfinite(days).all() or not np.isfinite(cutoff) or not np.isfinite(horizon) or horizon<=0:
        raise ValueError('Finite one-dimensional events and positive horizon required')
    eligible=bool(np.any((days>cutoff-horizon)&(days<=cutoff)))
    future=bool(np.any((days>cutoff)&(days<=cutoff+horizon)))
    return eligible,int(not future) if eligible else None

def visible_paths(edges,node_times,root,cutoff,hops=2):
    """Directed receptive field, preserving the original query cutoff every hop."""
    if root not in node_times or not np.isfinite(cutoff) or not isinstance(hops,int) or hops<0:
        raise ValueError('Valid root, cutoff and hop count required')
    if node_times[root] is not None and node_times[root]>cutoff:
        raise ValueError('Root is unavailable')
    seen={root};frontier={root}
    for _ in range(hops):
        reached={dst for src,dst in edges if src in frontier
                 and (node_times[dst] is None or node_times[dst]<=cutoff)}
        frontier=reached-seen;seen|=reached
    return sorted(seen)

def rank_auc(target,scores):
    """Positive-negative concordance with half credit for ties; O(n log n)."""
    y=np.asarray(target);s=np.asarray(scores,dtype=float)
    if y.ndim!=1 or s.shape!=y.shape or not np.isfinite(s).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Aligned binary labels and finite scores required')
    positives=int(y.sum());negatives=len(y)-positives
    if positives==0 or negatives==0:raise ValueError('Both classes required')
    order=np.argsort(s,kind='stable');s=s[order];y=y[order]
    starts=np.r_[0,1+np.flatnonzero(s[1:]!=s[:-1])];ends=np.r_[starts[1:],len(s)]
    group_pos=np.add.reduceat(y,starts);group_neg=ends-starts-group_pos
    lower_neg=np.cumsum(group_neg)-group_neg
    return float(np.sum(group_pos*(lower_neg+.5*group_neg))/(positives*negatives))

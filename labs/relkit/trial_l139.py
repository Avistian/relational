"""Load-bearing clinical-trial contracts. Times use one consistent unit."""
import numpy as np

def trial_target(start,analyses,cutoff,horizon=365):
    """Return (eligible, label); match released SQL, including modifier quirks."""
    if not np.isfinite(start) or not np.isfinite(cutoff) or horizon<=0:
        raise ValueError('finite start/cutoff and positive horizon required')
    if start>cutoff:return False,None
    values=[]
    for date,p,modifier,kind in analyses:
        if (kind=='Primary' and modifier!='>' and p is not None
                and 0<=p<=1 and cutoff<date<=cutoff+horizon):
            values.append(p)
    return (True,int(min(values)<=.05)) if values else (False,None)

def visibility_mask(times,owners,cutoffs):
    """Every sampled node inherits its own query cutoff, never a batch maximum."""
    times=np.asarray(times);owners=np.asarray(owners);cutoffs=np.asarray(cutoffs)
    if any(a.ndim!=1 for a in [times,owners,cutoffs]) or len(times)!=len(owners):
        raise ValueError('one time and owner per node required')
    if not np.isfinite(times).all() or not np.isfinite(cutoffs).all():
        raise ValueError('nonfinite temporal values')
    if len(owners) and (not np.issubdtype(owners.dtype,np.integer) or owners.min()<0 or owners.max()>=len(cutoffs)):
        raise ValueError('invalid query ownership')
    return times<=cutoffs[owners.astype(int)]

def keyed_auc(query_keys,targets,prediction_keys,scores):
    """Align complete unique (entity, cutoff) keys, then count tied rank wins."""
    q=[tuple(k) for k in query_keys];p=[tuple(k) for k in prediction_keys]
    if len(set(q))!=len(q) or len(set(p))!=len(p) or set(q)!=set(p):
        raise ValueError('keys must be unique and cover exactly the same queries')
    if len(targets)!=len(q) or len(scores)!=len(p):raise ValueError('length mismatch')
    mapping=dict(zip(p,scores));return rank_auc(targets,[mapping[k] for k in q])

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

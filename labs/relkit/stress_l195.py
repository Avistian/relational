"""Visible contracts for scoped falsification; complete saved predictions only."""
import math
import numpy as np


def paired_mae(truth, candidate, baseline):
    """Align full keys; positive baseline error minus candidate error favors candidate."""
    def index(rows):
        result={}
        for row in rows:
            if len(row)!=3:raise ValueError('Require entity, cutoff, value')
            entity,cutoff,value=row;key=(entity,cutoff)
            if key in result or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Duplicate key or invalid numeric value')
            result[key]=float(value)
        return result
    y=index(truth);a=index(candidate);b=index(baseline)
    if not y or y.keys()!=a.keys() or y.keys()!=b.keys():raise ValueError('Incomplete query identity')
    ae=[abs(y[k]-a[k]) for k in y];be=[abs(y[k]-b[k]) for k in y]
    return dict(keys=[list(k) for k in y],candidate_mae=math.fsum(ae)/len(y),baseline_mae=math.fsum(be)/len(y),advantage=[v-u for u,v in zip(ae,be)])


def interval_verdict(low, high, margin, complete=True):
    """Describe interval geometry; this is not an equivalence or causal test."""
    if type(complete) is not bool:raise ValueError('Completeness must be boolean')
    if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not math.isfinite(margin) or margin<0:
        raise ValueError('Finite nonnegative margin required')
    if not complete:return 'INCOMPLETE'
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in [low,high]) or low>high:
        raise ValueError('Finite ordered interval required')
    if low>margin:return 'ABOVE_MARGIN'
    if high<-margin:return 'BELOW_NEGATIVE_MARGIN'
    if margin>0 and low>=-margin and high<=margin:return 'WITHIN_MARGIN'
    return 'UNRESOLVED'


def claim_scope(claim, evidence):
    """Allowed conclusions for this saved-evidence packet, not a universal inference engine."""
    if claim=='pipeline':
        checks=['authenticated','complete','measured','comparable']
        return 'SCOPED_COMPARISON' if all(evidence.get(k) is True for k in checks) else 'INSUFFICIENT_EVIDENCE'
    if claim in ['relational_signal','architecture_cause','general_superiority','undervaluation']:return 'NOT_ESTABLISHED'
    if claim=='fresh_training':return 'NOT_RUN'
    raise ValueError('Unknown claim')


def keyed_auc(truth, predictions):
    """Rank AUROC with half credit for ties, aligned by (entity_id, cutoff)."""
    def index(rows):
        result={}
        for row in rows:
            if len(row)!=3:raise ValueError('Expected entity, cutoff, value')
            a,b,v=row;key=(a,b)
            if key in result or not math.isfinite(v):raise ValueError('Duplicate key or nonfinite value')
            result[key]=v
        return result
    labels=index(truth);probs=index(predictions)
    if not labels or labels.keys()!=probs.keys():raise ValueError('Incomplete query identity')
    if set(labels.values())!={0,1}:raise ValueError('Both binary classes required')
    if any(not 0<=v<=1 for v in probs.values()):raise ValueError('Probabilities outside [0,1]')
    ordered=sorted((probs[k],labels[k]) for k in labels)
    rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum+=((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
    npos=sum(labels.values());nneg=len(labels)-npos
    return (rank_sum-npos*(npos+1)/2)/(npos*nneg)


def cluster_interval(delta, entities, draws=2000, seed=137):
    """Resample whole drivers; retain row weighting inside every bootstrap draw."""
    d=np.asarray(delta,dtype=float);e=np.asarray(entities)
    if d.ndim!=1 or e.shape!=d.shape or not len(d) or not np.isfinite(d).all():raise ValueError('Invalid aligned losses')
    if draws<2:raise ValueError('Need at least two draws')
    groups,inverse=np.unique(e,return_inverse=True);n=len(groups)
    if n<2:return dict(status='INSUFFICIENT_ENTITIES',mean=float(d.mean()),entities=n,low=None,high=None)
    totals=np.bincount(inverse,weights=d);counts=np.bincount(inverse)
    rng=np.random.default_rng(seed);samples=rng.integers(0,n,size=(draws,n))
    estimates=totals[samples].sum(axis=1)/counts[samples].sum(axis=1)
    lo,hi=np.quantile(estimates,[.025,.975])
    return dict(status='CONDITIONAL_DESCRIPTIVE',mean=float(d.mean()),entities=n,low=float(lo),high=float(hi),draws=draws,seed=seed)

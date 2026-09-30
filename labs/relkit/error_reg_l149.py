"""Visible query-paired error analysis; original course extension, not paper code."""
import numpy as np

def paired_errors(keys, target, gnn_keys, gnn, fe_keys, fe):
    """Positive means the GNN has larger absolute error on the same query."""
    canonical=[tuple(k) for k in keys]
    y=np.asarray(target,dtype=float)
    if y.ndim!=1 or len(y)!=len(canonical) or not len(y) or not np.isfinite(y).all():
        raise ValueError('Require finite targets and complete query keys')
    if len(set(canonical))!=len(canonical):raise ValueError('Duplicate truth keys')
    aligned=[]
    for supplied,values in [(gnn_keys,gnn),(fe_keys,fe)]:
        supplied=[tuple(k) for k in supplied];v=np.asarray(values,dtype=float)
        if v.ndim!=1 or len(v)!=len(supplied) or not np.isfinite(v).all():raise ValueError('Invalid predictions')
        if len(set(supplied))!=len(supplied) or set(supplied)!=set(canonical):raise ValueError('Keys must be a bijection')
        lookup=dict(zip(supplied,v));aligned.append(np.array([lookup[k] for k in canonical]))
    return np.abs(aligned[0]-y)-np.abs(aligned[1]-y)

def nominate_slice(delta, entities, masks, split='val', min_rows=30, min_entities=10):
    """Choose largest positive supported mean on validation; report unsupported slices."""
    if split!='val':raise ValueError('Only validation may nominate a slice')
    d=np.asarray(delta,dtype=float);e=np.asarray(entities)
    if d.ndim!=1 or e.shape!=d.shape or not np.isfinite(d).all():raise ValueError('Invalid aligned losses')
    rows={}
    for name,mask in sorted(masks.items()):
        m=np.asarray(mask)
        if m.dtype!=bool or m.shape!=d.shape:raise ValueError('Require one boolean per query')
        n=int(m.sum());groups=len(np.unique(e[m]));supported=n>=min_rows and groups>=min_entities
        rows[name]=dict(rows=n,entities=groups,supported=supported,mean=float(d[m].mean()) if n else None)
    eligible=[name for name,r in rows.items() if r['supported'] and r['mean']>0]
    selected=min(eligible,key=lambda name:(-rows[name]['mean'],name)) if eligible else None
    return dict(selected=selected,slices=rows,split=split,min_rows=min_rows,min_entities=min_entities)

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

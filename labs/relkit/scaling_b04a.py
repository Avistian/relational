"""B04a visible course mechanisms. This is not a trained TabFlex model."""
import numpy as np

def linear_readout(q, k, v, eps=1e-6):
    """ELU+1 kernel readout using a d×value_width summary, never Q×S scores."""
    q,k,v=(np.asarray(x,dtype=np.float64) for x in (q,k,v))
    if any(x.ndim!=2 or 0 in x.shape or not np.isfinite(x).all() for x in (q,k,v)):
        raise ValueError('Finite nonempty matrices required')
    if q.shape[1]!=k.shape[1] or len(k)!=len(v) or not np.isfinite(eps) or eps<0:
        raise ValueError('Incompatible shapes or epsilon')
    pq=np.where(q>=0,q+1,np.exp(np.minimum(q,0)))
    pk=np.where(k>=0,k+1,np.exp(np.minimum(k,0)))
    summary=pk.T@v
    normalizer=pk.sum(axis=0)
    denominator=pq@normalizer+eps
    if (denominator<=0).any():raise ValueError('Degenerate kernel denominator')
    return (pq@summary)/denominator[:,None]

def fit_scale(support):
    """Population moments fitted only on support; constant columns use scale one."""
    x=np.asarray(support,dtype=np.float64)
    if x.ndim!=2 or 0 in x.shape or not np.isfinite(x).all():raise ValueError('Finite nonempty support required')
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1
    return mean,scale

def class_values(labels, classes, capacity):
    """Preserve declared output-column identities; never merge excess classes."""
    labels=np.asarray(labels);classes=np.asarray(classes)
    if labels.ndim!=1 or classes.ndim!=1 or not len(labels) or not len(classes):raise ValueError('Nonempty label vectors required')
    if len(np.unique(classes))!=len(classes) or len(classes)>capacity or not np.isin(labels,classes).all():raise ValueError('Unknown, duplicate or excess classes')
    return (labels[:,None]==classes[None,:]).astype(np.float64)

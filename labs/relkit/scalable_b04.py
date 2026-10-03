"""Learner mechanisms for B04; scalar attention is not the pretrained model."""
import numpy as np

def attention_readout(query, keys, values, scale=1.0):
    """One query, unmasked support keys: return value readout, weights, H/log(n)."""
    q,k,v=np.asarray(query,float),np.asarray(keys,float),np.asarray(values,float)
    if q.ndim!=1 or not q.size or k.ndim!=2 or v.ndim!=2 or not len(k) or k.shape[1]!=q.size or len(k)!=len(v):
        raise ValueError('Expected q[d], K[n,d], V[n,v] with n,d positive')
    if not all(np.isfinite(x).all() for x in (q,k,v)) or not np.isfinite(scale):
        raise ValueError('Attention inputs must be finite')
    logits=(k@q)*float(scale)/np.sqrt(q.size)
    if not np.isfinite(logits).all():raise ValueError('Attention logits overflowed')
    exp=np.exp(logits-logits.max());p=exp/exp.sum()
    positive=p[p>0];entropy=float(-(positive*np.log(positive)).sum()/np.log(len(p))) if len(p)>1 else 0.
    return p@v,p,entropy

def fit_missingness(support):
    """Support-only column means; a wholly missing column has frozen fill zero."""
    x=np.asarray(support,float)
    if x.ndim!=2 or not all(x.shape) or np.isinf(x).any():raise ValueError('Expected nonempty finite-or-NaN table')
    count=(~np.isnan(x)).sum(axis=0)
    return np.divide(np.nansum(x,axis=0),count,out=np.zeros(x.shape[1]),where=count>0)

def transform_missingness(rows, means, indicators=False):
    """No fitting: retain column order; append one binary flag per original column."""
    x,mu=np.asarray(rows,float),np.asarray(means,float)
    if x.ndim!=2 or mu.ndim!=1 or x.shape[1]!=mu.size or np.isinf(x).any() or not np.isfinite(mu).all():
        raise ValueError('Incompatible table or nonfinite means')
    missing=np.isnan(x);filled=np.where(missing,mu,x)
    return np.concatenate([filled,missing.astype(float)],axis=1) if indicators else filled

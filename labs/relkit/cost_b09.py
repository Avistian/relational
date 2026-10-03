"""B09 measurement contracts. All three axes of Pareto points are minimized."""
import numpy as np

def aligned_rmse(ids, truth, prediction_ids, predictions):
    """Authenticate full query identity before scoring; reject lost/duplicate rows."""
    ids=list(ids); prediction_ids=list(prediction_ids)
    truth=np.asarray(truth,dtype=float); predictions=np.asarray(predictions,dtype=float)
    if len(set(ids))!=len(ids) or len(set(prediction_ids))!=len(prediction_ids):
        raise ValueError('Duplicate query identity')
    if set(ids)!=set(prediction_ids) or truth.shape!=(len(ids),) or predictions.shape!=(len(ids),):
        raise ValueError('Query identity or shape mismatch')
    if not len(ids) or not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError('Empty or nonfinite prediction packet')
    lookup=dict(zip(prediction_ids,predictions))
    errors=np.array([lookup[k] for k in ids])-truth
    return float(np.sqrt(np.mean(errors**2)))

def pareto_front(points):
    """Keep points with no weakly better competitor that is strictly better somewhere."""
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or not np.isfinite(p).all():raise ValueError('Finite matrix required')
    return np.array([not np.any(np.all(p<=x,axis=1)&np.any(p<x,axis=1)) for x in p])

def support_attention(query, keys, values, support_count):
    """Illustrative scaled dot-product attention, not EXAONE SSMax or a checkpoint."""
    q=np.asarray(query,float);k=np.asarray(keys,float);v=np.asarray(values,float)
    if not 0<support_count<=len(k) or len(k)!=len(v):raise ValueError('Invalid support boundary')
    logits=q@k[:support_count].T/np.sqrt(q.shape[-1])
    logits-=logits.max(axis=-1,keepdims=True)
    weights=np.exp(logits);weights/=weights.sum(axis=-1,keepdims=True)
    return weights@v[:support_count]

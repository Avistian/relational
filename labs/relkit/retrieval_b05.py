"""B05 course mechanisms, deliberately distinct from the released pretraining sampler."""
import numpy as np

def feature_view(table, target):
    """Remove the target before any distance fitting; return independent arrays."""
    table=np.asarray(table,dtype=float)
    if table.ndim!=2 or table.shape[1]<2 or not 0<=target<table.shape[1]:
        raise ValueError('A valid target and at least one feature are required')
    if not np.isfinite(table).all():raise ValueError('Course input must be finite')
    return np.delete(table,target,axis=1),table[:,target].copy()

def neighbors(support, query, ids, k):
    """Return row IDs, not array offsets. Statistics are fit on eligible support only."""
    support=np.asarray(support,dtype=float);query=np.asarray(query,dtype=float);ids=np.asarray(ids)
    if support.ndim!=2 or query.ndim!=2 or support.shape[1]!=query.shape[1]:raise ValueError('Shape mismatch')
    if len(ids)!=len(support) or len(set(ids.tolist()))!=len(ids) or not 1<=k<=len(ids):raise ValueError('Invalid IDs or k')
    if not np.isfinite(support).all() or not np.isfinite(query).all():raise ValueError('Nonfinite feature')
    scale=support.std(axis=0);scale=np.where(scale>0,scale,1.)
    # Center cancels in pairwise differences; scale is still support-only.
    distances=(((query[:,None,:]-support[None,:,:])/scale)**2).sum(axis=2)
    return np.stack([ids[np.lexsort((ids,d))[:k]] for d in distances])

def episode(table, target, anchor, size, support_n, seed):
    """Paper-order teaching episode: remove target, retrieve, then partition once."""
    x,y=feature_view(table,target)
    if not 0<=anchor<len(x) or not 0<support_n<size<=len(x):raise ValueError('Invalid episode sizes')
    selected=neighbors(x,x[[anchor]],np.arange(len(x)),size)[0]
    order=np.random.default_rng(seed).permutation(selected);s=order[:support_n];q=order[support_n:]
    return dict(selected_ids=selected.tolist(),support_ids=s.tolist(),query_ids=q.tolist(),support_x=x[s].tolist(),query_x=x[q].tolist(),support_y=y[s].tolist(),query_y=y[q].tolist())

def overlap_status(train, test):
    """Metadata can flag overlap; absence of a flag cannot certify independence."""
    if any(train.get(k) is not None and train.get(k)==test.get(k) for k in ('dataset_id','sha256')):return 'OVERLAP'
    if train.get('family') not in (None,'unknown') and train.get('family')==test.get('family'):return 'RELATED'
    if train.get('name') and train.get('name')==test.get('name'):return 'REVIEW_NAME'
    return 'NOT_ESTABLISHED'

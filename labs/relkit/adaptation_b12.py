"""B12 fixed-kernel course diagnostic. Not a Griffin/OpenRFM/Kumo model.

All arithmetic is binary64. Queries are Q×D; supports S×D; label values S.
No optimizer, learned checkpoint, benchmark or paper-performance claim.
"""
import numpy as np

def eligible_support(event, available, horizon, cutoff):
    """A label is usable once its event, arrival and outcome window are complete."""
    event,available,horizon=np.broadcast_arrays(np.asarray(event,float),np.asarray(available,float),np.asarray(horizon,float))
    if not np.isfinite(cutoff) or np.any(horizon<0):raise ValueError('Finite cutoff and nonnegative horizons required')
    return np.isfinite(event)&np.isfinite(available)&np.isfinite(horizon)&(event<=cutoff)&(available<=cutoff)&(event+horizon<=cutoff)

def label_attention(query, keys, labels, allowed):
    """Masked softmax label read; declared probability .5 when no labels admitted.

    Missing forbidden values are removed BEFORE multiplication: 0*NaN is NaN.
    """
    query=np.asarray(query,float);keys=np.asarray(keys,float);labels=np.asarray(labels,float);allowed=np.asarray(allowed)
    if query.ndim!=2 or keys.ndim!=2 or query.shape[1]!=keys.shape[1] or not query.shape[1]:raise ValueError('Q/K widths must match')
    if labels.shape!=(len(keys),) or allowed.shape!=(len(query),len(keys)) or allowed.dtype!=bool:raise ValueError('Label/mask shapes or mask type invalid')
    if not np.isfinite(query).all() or not np.isfinite(keys).all():raise ValueError('Finite features required')
    visible=allowed.any(axis=0)
    if not np.isfinite(labels[visible]).all() or np.any((labels[visible]<0)|(labels[visible]>1)):raise ValueError('Visible labels must be probabilities')
    result=np.full(len(query),.5)
    active=allowed.any(axis=1)
    if active.any():
        scores=(query[active]@keys.T)/np.sqrt(query.shape[1])
        scores=np.where(allowed[active],scores,-np.inf)
        weights=np.exp(scores-scores.max(axis=1,keepdims=True))
        weights/=weights.sum(axis=1,keepdims=True)
        result[active]=weights@np.where(visible,labels,0.)
    return result

def intervene(labels, mode, permutation):
    """Change only support labels; preserve their multiset for the shuffle arm."""
    labels=np.asarray(labels,float);permutation=np.asarray(permutation)
    if labels.ndim!=1 or permutation.dtype.kind not in 'iu' or not np.array_equal(np.sort(permutation),np.arange(len(labels))):raise ValueError('A full bijective permutation is required')
    if mode=='intact':return labels.copy()
    if mode=='shuffled':return labels[permutation].copy()
    if mode=='hidden':return np.full_like(labels,np.nan)
    raise ValueError('Unknown label intervention')

def make_fixture(seed):
    """Eight support rows, twelve queries, each joined to one numeric parent row.

    The parent feature affects the row embedding. Separate membership edges control
    whether each query's walk reaches zero or four support task rows. All eight
    support rows are globally available at cutoff10. No label enters the features.
    """
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(20,4));parent=rng.normal(size=(20,4))
    weights=np.eye(4)+rng.normal(scale=.12,size=(4,4))
    labels=((x+.25*parent)[:,0]>0).astype(float)
    high=np.zeros((12,8),bool)
    for i in range(12):high[i,[(i+j)%8 for j in range(4)]]=True
    return dict(x=x,parent=parent,weights=weights,labels=labels,high=high,low=np.zeros((12,8),bool),
                event=np.arange(8,dtype=float),available=np.arange(8,dtype=float)+2,horizon=np.ones(8),
                permutation=rng.permutation(8),keys=[(i,10) for i in range(8,20)])

def predict(fixture, reachability, mode, channel):
    """Visible two-stage course computation; TODO functions are called here.

    1. Parent feature aggregation + fixed projection creates row embeddings Z.
    2. Relation channel reads only reachable support labels.
    3. Dual channel also reads all eligible support labels through Z similarities.
       Equal averaging of the two outputs is a COURSE rule, not OpenRFM fusion.
    """
    if reachability not in ('high','low') or channel not in ('relational','dual'):raise ValueError('Unknown condition')
    z=(fixture['x']+.25*fixture['parent'])@fixture['weights']
    labels=intervene(fixture['labels'][:8],mode,fixture['permutation'])
    eligible=eligible_support(fixture['event'],fixture['available'],fixture['horizon'],10.)&np.isfinite(labels)
    relation_mask=fixture[reachability]&eligible[None,:]
    relational=label_attention(z[8:],z[:8],labels,relation_mask)
    if channel=='relational':return relational
    batch_mask=np.broadcast_to(eligible,(12,8))
    batch=label_attention(z[8:],z[:8],labels,batch_mask)
    return .5*(relational+batch)

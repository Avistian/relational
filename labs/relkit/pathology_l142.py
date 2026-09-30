"""Live aggregation, pair-specific fusion and keyed paired error contracts."""
import torch,numpy as np

def edge_sum(values,destination,num_destinations):
    """Sum incoming messages without mixing destination rows or head axes."""
    output=values.new_zeros((num_destinations,)+tuple(values.shape[1:]))
    output.index_add_(0,destination,values)
    return output

def route_fuse(source,fact,edges,source_transform=None,fact_transform=None):
    """Aggregate only this source role; apply source bias once per fact."""
    gathered=edge_sum(source[edges[0]],edges[1],len(fact))
    left=gathered if source_transform is None else source_transform(gathered)
    right=fact if fact_transform is None else fact_transform(fact)
    return left+right

def paired_loss_gap(keys,targets,composite_keys,composite,ordinary_keys,ordinary):
    """Return ordinary minus composite absolute error, aligned by full query key."""
    q=[tuple(k) for k in keys];y=np.asarray(targets,dtype=float)
    if not q or len(set(q))!=len(q) or y.shape!=(len(q),) or not np.isfinite(y).all():
        raise ValueError('Require unique keys and finite targets')
    aligned=[]
    for kk,pp in [(composite_keys,composite),(ordinary_keys,ordinary)]:
        k=[tuple(x) for x in kk];p=np.asarray(pp,dtype=float)
        if len(set(k))!=len(k) or set(k)!=set(q) or p.shape!=(len(k),) or not np.isfinite(p).all():
            raise ValueError('Require the same complete unique query population')
        lookup=dict(zip(k,p));aligned.append(np.array([lookup[x] for x in q]))
    return np.abs(aligned[1]-y)-np.abs(aligned[0]-y)

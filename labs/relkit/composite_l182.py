"""Visible L182 mechanisms. Single-head paper equation, not a complete trained RelGNN."""
import numpy as np

def legal_fusion(source, bridge, source_index, destination_index, times, cutoffs, W_source, W_bridge):
    """One FK-resolved source per bridge; retain events strictly before each owner cutoff."""
    source=np.asarray(source,float);bridge=np.asarray(bridge,float)
    si=np.asarray(source_index);di=np.asarray(destination_index)
    times=np.asarray(times,float);cutoffs=np.asarray(cutoffs,float)
    ws=np.asarray(W_source,float);wm=np.asarray(W_bridge,float)
    if source.ndim!=2 or bridge.ndim!=2 or ws.ndim!=2 or wm.ndim!=2:raise ValueError('Matrices required')
    n=len(bridge)
    if si.shape!=(n,) or di.shape!=(n,) or times.shape!=(n,) or cutoffs.ndim!=1:raise ValueError('Aligned route rows required')
    if si.dtype.kind not in 'iu' or di.dtype.kind not in 'iu':raise ValueError('Integer FK indices required')
    if np.any(si<0) or np.any(si>=len(source)) or np.any(di<0) or np.any(di>=len(cutoffs)):raise ValueError('Unresolved foreign key')
    if ws.shape[0]!=source.shape[1] or wm.shape[0]!=bridge.shape[1] or ws.shape[1]!=wm.shape[1]:raise ValueError('Projection shapes differ')
    if not all(np.isfinite(x).all() for x in [source,bridge,times,cutoffs,ws,wm]):raise ValueError('Finite inputs required')
    keep=times<cutoffs[di]
    return source[si[keep]]@ws+bridge[keep]@wm,di[keep]

def attend(query, messages, destination):
    """One head with identity Q/K/V projections; zero for an empty neighborhood."""
    query=np.asarray(query,float);messages=np.asarray(messages,float);destination=np.asarray(destination)
    if query.ndim!=2 or messages.ndim!=2 or query.shape[1]!=messages.shape[1] or query.shape[1]<1:raise ValueError('Matching positive widths required')
    if destination.shape!=(len(messages),) or destination.dtype.kind not in 'iu' or np.any(destination<0) or np.any(destination>=len(query)):raise ValueError('Invalid destination')
    if not np.isfinite(query).all() or not np.isfinite(messages).all():raise ValueError('Finite input required')
    result=np.zeros_like(query);weights=np.zeros(len(messages))
    for i,q in enumerate(query):
        idx=np.flatnonzero(destination==i)
        if len(idx):
            logits=messages[idx]@q/np.sqrt(query.shape[1]);w=np.exp(logits-logits.max());w/=w.sum()
            result[i]=w@messages[idx];weights[idx]=w
    return result,weights

def keyed_auc(truth_keys, labels, prediction_keys, probabilities):
    """Pairwise AUROC with half credit for ties, after complete-key alignment."""
    q=[tuple(x) for x in truth_keys];p=[tuple(x) for x in prediction_keys]
    y=np.asarray(labels,float);v=np.asarray(probabilities,float)
    if not q or any(len(k)!=2 for k in q+p) or len(set(q))!=len(q) or len(set(p))!=len(p) or set(q)!=set(p):raise ValueError('Complete unique entity/cutoff keys required')
    if y.shape!=(len(q),) or v.shape!=(len(p),) or not np.isfinite(y).all() or not np.isfinite(v).all() or set(y)!={0.,1.} or np.any((v<0)|(v>1)):raise ValueError('Binary truth and finite probabilities required')
    lookup=dict(zip(p,v));ordered=np.array([lookup[k] for k in q])
    delta=ordered[y==1,None]-ordered[y==0][None,:]
    return float(np.mean((delta>0)+.5*(delta==0)))

def factorial_interaction(scores):
    """Paired seed contrast: (relational+composite minus relational+plain) minus the single-prior difference."""
    x=np.asarray(scores,float)
    if x.ndim!=2 or x.shape[1]!=4 or x.shape[0]<2 or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('At least two paired rows and four AUROC arms required')
    return (x[:,3]-x[:,2])-(x[:,1]-x[:,0])

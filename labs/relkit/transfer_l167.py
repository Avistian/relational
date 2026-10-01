"""Visible mechanisms for the tabular-to-relational transfer lesson."""

def temporal_summary(queries, events):
    """Course fixture: [entity,cutoff] queries; [entity,event_time,value] events.

    Return [count,mean] for matching events strictly before each owner cutoff.
    Event time is assumed to equal availability time ONLY in this fixture.
    """
    import numpy as np
    q=np.asarray(queries);e=np.asarray(events)
    if q.ndim!=2 or q.shape[1]!=2 or e.ndim!=2 or e.shape[1]!=3:
        raise ValueError('Expected query pairs and event triples')
    if not np.isfinite(q).all() or not np.isfinite(e).all():
        raise ValueError('Nonfinite fixture input')
    if len(set(map(tuple,q)))!=len(q):raise ValueError('Duplicate query keys')
    result=[]
    for entity,cutoff in q:
        values=e[(e[:,0]==entity)&(e[:,1]<cutoff),2]
        result.append([len(values),float(values.mean()) if len(values) else 0.])
    return np.asarray(result,dtype=float).reshape(-1,2)

def eligible_support(keys, label_ready, cutoff):
    """Return indices whose query time AND label availability precede cutoff."""
    import numpy as np
    k=np.asarray(keys);ready=np.asarray(label_ready)
    if k.ndim!=2 or k.shape[1]!=2 or ready.shape!=(len(k),):
        raise ValueError('Invalid support shape')
    if not np.isfinite(k).all() or not np.isfinite(ready).all() or not np.isfinite(cutoff):
        raise ValueError('Nonfinite support input')
    if len(set(map(tuple,k)))!=len(k) or np.any(ready<k[:,1]):
        raise ValueError('Duplicate keys or labels ready before prediction time')
    return np.flatnonzero((k[:,1]<cutoff)&(ready<cutoff))

def keyed_auc(expected_keys, labels, prediction_keys, probabilities):
    """Align exact composite identities; AUROC from average ranks (ties=half)."""
    import numpy as np
    expected=np.asarray(expected_keys);got=np.asarray(prediction_keys)
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    for keys in (expected,got):
        if keys.ndim!=2 or keys.shape[1]!=2 or len(set(map(tuple,keys)))!=len(keys):
            raise ValueError('Duplicate or malformed composite keys')
    if y.shape!=(len(expected),) or p.shape!=(len(got),) or set(y)!={0,1}:
        raise ValueError('Invalid label/prediction population')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid class-1 probabilities')
    lookup=dict(zip(map(tuple,got),p))
    if set(lookup)!=set(map(tuple,expected)):raise ValueError('Incomplete or extra query keys')
    scores=np.array([lookup[tuple(key)] for key in expected])
    order=np.argsort(scores,kind='stable');sorted_scores=scores[order]
    ranks=np.empty(len(scores),dtype=float);i=0
    while i<len(scores):
        j=i+1
        while j<len(scores) and sorted_scores[j]==sorted_scores[i]:j+=1
        ranks[order[i:j]]=(i+1+j)/2.;i=j
    positives=int(y.sum());negatives=len(y)-positives
    return float((ranks[y==1].sum()-positives*(positives+1)/2)/(positives*negatives))

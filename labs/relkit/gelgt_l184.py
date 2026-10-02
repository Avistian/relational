"""Visible course mechanisms. This is not a repaired full GelGT trainer."""
from collections import deque
import numpy as np

def temporal_bfs(edges, times, seed, cutoff, budget):
    """Strict prior-time, sorted two-hop BFS; seed counts toward budget.

    Untimed rows are excluded in this course mechanism. The release uses <=
    and admits untimed rows, so this policy is an explicit course deviation.
    """
    if budget < 1 or not 0 <= seed < len(times) or not np.isfinite(cutoff):
        raise ValueError('invalid sampling contract')
    adj=[set() for _ in times]
    for a,b in edges:
        if not (0 <= a < len(times) and 0 <= b < len(times)):
            raise ValueError('invalid edge')
        adj[a].add(b);adj[b].add(a)
    out=[seed];seen={seed};queue=deque([(seed,0)])
    while queue and len(out)<budget:
        node,hop=queue.popleft()
        if hop==2:continue
        for v in sorted(adj[node]):
            if v in seen or times[v] is None or not np.isfinite(times[v]) or times[v]>=cutoff:continue
            seen.add(v);out.append(v);queue.append((v,hop+1))
            if len(out)==budget:break
    return out

def gaussian_features(lags, center, width):
    """exp(-((lag-center)/width)^2), the release's radial basis kernel.

    width is the effective positive width (release uses abs(raw)+1e-5).
    A learned projection mixes kernels into per-head attention biases.
    """
    x=np.asarray(lags,dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(center) or not np.isfinite(width) or width<=0:
        raise ValueError('finite inputs and positive width required')
    return np.exp(-((x-center)/width)**2)

def keyed_mae(truth, predictions):
    """Rows are (entity, cutoff, value); order must not affect scoring."""
    def index(rows):
        d={}
        for entity,cutoff,value in rows:
            k=(entity,cutoff)
            if entity is None or cutoff is None or k in d or not np.isfinite(value):
                raise ValueError('duplicate/missing/nonfinite query')
            d[k]=float(value)
        if not d:raise ValueError('empty population')
        return d
    y=index(truth);p=index(predictions)
    if y.keys()!=p.keys():raise ValueError('query populations differ')
    return sum(abs(y[k]-p[k]) for k in y)/len(y)

def semantic_refine(embeddings, hops, keep):
    """Course policy: preserve seed + all one-hop nodes, rank distant nodes.

    Reject impossible protected budgets rather than dropping the seed in ties.
    Stable score ties use original index; return indices in original order.
    """
    x=np.asarray(embeddings,float);h=np.asarray(hops)
    if x.ndim!=2 or len(x)!=len(h) or not np.isfinite(x).all() or h[0]!=0:
        raise ValueError('invalid tokens')
    protected=np.flatnonzero(h<=1).tolist()
    if not len(protected)<=keep<=len(x):raise ValueError('protected nodes exceed budget')
    scores=x@x[0]
    candidates=sorted((i for i in range(len(x)) if h[i]==2),key=lambda i:(-scores[i],i))
    selected=protected+candidates[:keep-len(protected)]
    if len(selected)!=keep:raise ValueError('insufficient real candidates')
    return sorted(selected)

def attention_trace(lags, center=2., width=2., projection=1.):
    """Synthetic single-head example with equal QK logits, values [1,3,9]."""
    bias=projection*gaussian_features(lags,center,width)
    weights=np.exp(bias-bias.max());weights/=weights.sum()
    return {'bias':bias.tolist(),'weights':weights.tolist(),
            'output':float(weights@np.array([1.,3.,9.]))}

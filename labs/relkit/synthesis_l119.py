"""Visible course mechanisms; none is a substitute for the RelBench experiment."""
# %% Shared dependencies (PROVIDED)
from collections import Counter, defaultdict
import math
import numpy as np

# %% Task 1: accuracy ceiling for an observed representation
def collision_ceiling(features, labels):
    """Best deterministic accuracy on THESE observations, not a population bound."""
    if not len(labels) or len(features) != len(labels):
        raise ValueError('Need equally many nonempty features and labels')
    groups = defaultdict(Counter)
    for row, label in zip(features, labels):
        groups[tuple(row)][label] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(labels)

# %% Task 2: preserve the meaning of each relationship
def typed_sum(values, edges, weights):
    """Scalar relation-specific messages, reduced at the destination node."""
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    for source, destination, relation in edges:
        out[destination] += weights[relation] * values[source]
    return out

# %% Task 3: preserve one cutoff across a graph walk
def eligible_nodes(edges, event_times, available_times, seed, cutoff, hops):
    """Undirected course walk; both clocks known, None explicitly timeless."""
    if len(event_times) != len(available_times) or hops < 0:
        raise ValueError('Invalid clocks or hop count')
    def legal(i):
        return all(t is None or t <= cutoff for t in (event_times[i], available_times[i]))
    if not legal(seed):
        raise ValueError('Seed is not available at query time')
    seen = {seed}
    for _ in range(hops):
        additions = set()
        for u, v in edges:
            if u in seen and legal(v): additions.add(v)
            if v in seen and legal(u): additions.add(u)
        seen |= additions
    return sorted(seen)

# %% Task 4: score predictions by query identity
def aligned_mae(pred_ids, predictions, target_ids, targets):
    """Require a one-to-one query join before computing mean absolute error."""
    if (not len(targets) or len(pred_ids) != len(predictions)
        or len(target_ids) != len(targets)
        or len(set(pred_ids)) != len(pred_ids)
        or len(set(target_ids)) != len(target_ids)
        or set(pred_ids) != set(target_ids)):
        raise ValueError('Query identities must be unique and cover the same population')
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError('Nonfinite values')
    lookup = dict(zip(pred_ids, predictions))
    return math.fsum(abs(float(lookup[key]) - float(y))
                     for key, y in zip(target_ids, targets)) / len(targets)

# %% Mechanism fixtures (PROVIDED; not a benchmark)
def course_experiment():
    amounts = np.array([[10.,30.,50.],[50.,30.,10.]])
    times = np.array([1.,2.,3.])
    coarse = [(len(x),sum(x),np.mean(x),max(x)) for x in amounts]
    time_weighted = [(float(np.dot(times,x)),) for x in amounts]
    labels = [1,0]  # Authored rising/falling labels; no inferred customer outcome.
    cycle = [(i,(i+1)%6,'r') for i in range(6)]
    triangles = [(0,1,'r'),(1,2,'r'),(2,0,'r'),(3,4,'r'),(4,5,'r'),(5,3,'r')]
    def undirected(edges): return edges + [(v,u,r) for u,v,r in edges]
    states=[]
    for edges in [cycle,triangles]:
        x=np.ones(6);history=[x.tolist()]
        for _ in range(4):
            x=x+typed_sum(x,undirected(edges),{'r':1.})
            history.append(x.tolist())
        states.append(history)
    typed=[typed_sum([10.,4.,0.],[(0,2,a),(1,2,b)],{'buy':1.,'refund':-1.})[2]
           for a,b in [('buy','refund'),('refund','buy')]]
    eligible=eligible_nodes([(0,1),(1,2),(1,3),(1,4)],
                            [None,None,5,6,11],[None,None,9,6,11],0,7,2)
    return dict(status='PASS',scope='COURSE_ONLY',coarse=coarse,
                coarse_ceiling=collision_ceiling(coarse,labels),
                augmented_ceiling=collision_ceiling(time_weighted,labels),
                time_weighted=time_weighted,typed_outputs=typed,
                untyped_outputs=[14.,14.],cycle_states=states[0],triangle_states=states[1],
                indistinguishable=states[0]==states[1],eligible_nodes=eligible)

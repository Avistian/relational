"""Learner contracts used by preparation, checkpoint selection and final scoring."""
import random
import numpy as np

def select_config(configs, validation):
    scores=np.asarray(validation,dtype=float)
    if not configs or scores.shape!=(len(configs),) or not np.isfinite(scores).all():
        raise ValueError('One finite validation score is required for every candidate')
    return configs[int(np.argmin(scores))]

def paired_errors(keys, target, a_keys, a_pred, b_keys, b_pred):
    keys=[tuple(k) for k in keys]
    if len(set(keys))!=len(keys) or not keys:raise ValueError('Reference keys must be unique')
    def aligned(ks,ps):
        ks=[tuple(k) for k in ks];ps=np.asarray(ps,dtype=float)
        if len(ks)!=len(set(ks)) or set(ks)!=set(keys) or ps.shape!=(len(ks),) or not np.isfinite(ps).all():
            raise ValueError('Predictions must cover exactly the unique reference queries')
        lookup=dict(zip(ks,ps));return np.array([lookup[k] for k in keys])
    y=np.asarray(target,dtype=float)
    if y.shape!=(len(keys),) or not np.isfinite(y).all():raise ValueError('Invalid targets')
    return np.abs(aligned(a_keys,a_pred)-y)-np.abs(aligned(b_keys,b_pred)-y)

def temporal_context(root, cutoff, adjacency, times, k, seed):
    if k<1:raise ValueError('Context must include root')
    if times[root] is not None and times[root]>cutoff:raise ValueError('Future root')
    if k==1:return [root]
    rng=random.Random(seed);chosen=[root];seen={root};frontier=[root]
    # Two-hop BFS keeps selected connectors. Only legal nodes enter the frontier.
    for depth in range(2):
        next_frontier=[]
        for node in frontier:
            neighbors=sorted(adjacency[node]);rng.shuffle(neighbors)
            for other in neighbors:
                if other in seen or (times[other] is not None and times[other]>cutoff):continue
                seen.add(other);chosen.append(other);next_frontier.append(other)
                if len(chosen)==k:return chosen
        frontier=next_frontier
    # Explicit legal padding; never a global fallback. Repeated identities retain all slots.
    return chosen+rng.choices(chosen,k=k-len(chosen))

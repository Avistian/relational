"""Learner contracts, used by the complete ContextGNN lane."""
import numpy as np
import torch

def fuse_scores(tower, local, owners, items, local_offset):
    """Replace local entries; never add local evidence to a retained tower score."""
    if tower.ndim!=2 or len(local)!=len(owners) or len(items)!=len(local):
        raise ValueError('Mismatched score shapes')
    out=tower.clone()
    out[owners,items]=local+local_offset[owners]
    return out

def audit_cutoffs(times, owners, cutoffs):
    times=np.asarray(times);owners=np.asarray(owners);cutoffs=np.asarray(cutoffs)
    if len(times)!=len(owners) or np.any(owners<0) or np.any(owners>=len(cutoffs)):
        raise ValueError('Invalid owner mapping')
    if np.any(times>cutoffs[owners]):raise ValueError('Future event entered a query')
    return len(times)

def keyed_map(keys, targets, prediction_keys, predictions, k):
    """RelBench AP denominator=min(number of distinct positives,k), then mean."""
    keys=list(map(tuple,keys));prediction_keys=list(map(tuple,prediction_keys));p=np.asarray(predictions)
    if len(set(keys))!=len(keys) or len(set(prediction_keys))!=len(prediction_keys) or set(keys)!=set(prediction_keys):
        raise ValueError('Duplicate, missing or mismatched query identities')
    if k<1 or p.shape!=(len(keys),k) or len(targets)!=len(keys) or not len(keys):
        raise ValueError('Invalid population/ranking shape')
    lookup=dict(zip(prediction_keys,p));aps=[]
    for key,ys in zip(keys,targets):
        ys=set(map(int,ys));ranking=lookup[key]
        if len(set(ranking))!=k or not ys:raise ValueError('Duplicate recommendation or empty target')
        hits=np.array([int(v in ys) for v in ranking])
        aps.append(float((hits*np.cumsum(hits)/np.arange(1,k+1)).sum()/min(len(ys),k)))
    return float(np.mean(aps))

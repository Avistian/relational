"""Three learner-owned contracts used by the model, cache audit and trainer."""
import math
import torch

def mix_five(parts,projection):
    """Project five equally shaped B x K x d token components in source order."""
    if len(parts)!=5 or any(p.shape!=parts[0].shape for p in parts):
        raise ValueError('Five equally shaped components required')
    return projection(torch.cat(parts,dim=-1))

def audit_tokens(keys,timestamps):
    """Return (query row, token slot) future violations; None means undated."""
    if len(set(keys))!=len(keys) or len(keys)!=len(timestamps):
        raise ValueError('Unique entity/cutoff keys and matching rows required')
    return [(i,j) for i,((entity,cutoff),row) in enumerate(zip(keys,timestamps))
            for j,stamp in enumerate(row) if stamp is not None and stamp>cutoff]

def last_validation_min(scores):
    """Released <= checkpoint rule; ties replace the previous checkpoint."""
    if not scores or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Nonempty finite validation history required')
    return min(range(len(scores)),key=lambda i:(scores[i],-i))

def keyed_mae(keys,targets,pred_keys,predictions):
    import numpy as np
    if len(set(keys))!=len(keys) or len(set(pred_keys))!=len(pred_keys) or set(keys)!=set(pred_keys):
        raise ValueError('Prediction query identity mismatch')
    if len(targets)!=len(keys) or len(predictions)!=len(pred_keys):raise ValueError('Length mismatch')
    lookup=dict(zip(pred_keys,predictions));p=np.array([lookup[k] for k in keys]);y=np.asarray(targets)
    if not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Nonfinite metric')
    return float(np.abs(p-y).mean())

"""B07a learner-owned operations; integrated into released HyperFast inference."""
import math
import numpy as np
import torch

def class_weights(per_sample, hidden, labels, n_classes):
    """Return (hidden+1, classes); last row is the generated bias."""
    if len(labels)==0 or per_sample.shape!=(len(labels),hidden.shape[1]+1):
        raise ValueError('Incompatible nonempty support shapes')
    rows=[]
    for label in range(n_classes):
        mask=labels==label
        if not bool(mask.any()):mask=torch.ones_like(labels,dtype=torch.bool)
        row=per_sample[mask].mean(0).clone()
        row[:-1]+=hidden[mask].mean(0)
        rows.append(row)
    return torch.stack(rows).T

def retrieval_bias(logits, query, support, labels, amount):
    """Euclidean1-NN correction, first support row wins ties; do not mutate logits."""
    if len(support)==0 or len(support)!=len(labels):raise ValueError('Empty or mismatched support')
    result=logits.clone()
    for start in range(0,len(query),128):
        distance=torch.linalg.vector_norm(query[start:start+128,None,:]-support[None,:,:],ord=2,dim=2)
        nearest=labels[distance.argmin(1)]
        result[torch.arange(start,min(start+128,len(query))),nearest]+=amount
    return result

def break_even(build_a, per_a, build_b, per_b, refreshes=0):
    """First nonnegative integer Q where A is strictly cheaper, or None.

    Each refresh reconstructs BOTH methods. If A starts cheaper but later loses,
    this is only the first winning Q, not a claim of permanent dominance.
    """
    costs=[build_a,per_a,build_b,per_b]
    if not all(math.isfinite(v) and v>=0 for v in costs) or not isinstance(refreshes,int) or refreshes<0:
        raise ValueError('Costs must be finite/nonnegative; refresh count a nonnegative integer')
    offset=(refreshes+1)*(build_a-build_b)
    saving=per_b-per_a
    if offset<0:return 0
    if saving<=0:return None
    return math.floor(offset/saving)+1

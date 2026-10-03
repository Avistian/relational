"""Live B11 learner operations."""
import math
import torch

def eligible(event, arrival, cutoff):
    """Strict course policy: both clocks must be known and no later than cutoff."""
    return torch.isfinite(event) & torch.isfinite(arrival) & (event<=cutoff) & (arrival<=cutoff)

def explicit_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False):
    """Visible scaled dot-product attention; empty allowed sets return zero.

    B11 uses zero attention dropout, matching the selected source block setting.
    A boolean mask says allowed; a floating mask adds a score bias.
    """
    if dropout_p or is_causal:
        raise ValueError('B11 checks noncausal attention with zero attention dropout')
    scores=q @ k.transpose(-1,-2) / math.sqrt(q.shape[-1])
    if attn_mask is not None:
        scores=scores.masked_fill(~attn_mask,-torch.inf) if attn_mask.dtype==torch.bool else scores+attn_mask
    empty=torch.isneginf(scores).all(-1,keepdim=True)
    weights=torch.softmax(scores.masked_fill(empty,0.),dim=-1).masked_fill(empty,0.)
    return weights @ v

def first_validation_min(scores):
    """Return zero-based first best epoch; test scores cannot enter this API."""
    if not scores or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Require a nonempty finite validation history')
    return min(range(len(scores)),key=lambda i:scores[i])

"""L057 visible OOF stack and Caruana-style greedy convex ensemble, binary only."""
import numpy as np


def train_indices(fold_ids, heldout):
    """Rows eligible for fitting an OOF base model; no held-out labels enter fit."""
    return np.flatnonzero(np.asarray(fold_ids) != heldout)


def write_oof(matrix, seen, indices, predictions):
    """Scatter by original row position; reject missing/duplicate/nonfinite predictions."""
    indices=np.asarray(indices); predictions=np.asarray(predictions)
    if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or len(np.unique(indices))!=len(indices):
        raise ValueError('OOF indices must be unique integers')
    if np.any(indices<0) or np.any(indices>=len(matrix)) or np.any(seen[indices]!=0):
        raise ValueError('OOF row covered twice or outside row universe')
    if predictions.shape!=(len(indices),matrix.shape[1]) or not np.isfinite(predictions).all() or np.any((predictions<0)|(predictions>1)):
        raise ValueError('Expected aligned finite probabilities')
    matrix[indices]=predictions
    seen[indices]+=1


def blend(predictions, weights):
    """[rows, models] @ [models] -> positive-class probability [rows]."""
    p=np.asarray(predictions,float); w=np.asarray(weights,float)
    if p.ndim!=2 or w.shape!=(p.shape[1],) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid probability matrix')
    if not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1):
        raise ValueError('Weights must be finite, nonnegative and sum to one')
    return p@w


def binary_loss(y, p):
    """Binary cross-entropy (natural logs); clip only for numerical stability."""
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all():
        raise ValueError('Expected aligned binary targets and predictions')
    p=np.clip(p,1e-7,1-1e-7)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def greedy_select(oof, y, steps=40):
    """Caruana §2: selection with replacement, retaining the best prefix.

    Ties choose the earliest column/prefix deterministically; no sorted initialization
    or bagged libraries. Test labels are deliberately absent from the interface.
    """
    p=np.asarray(oof,float)
    if steps<1 or int(steps)!=steps:raise ValueError('Positive integer steps required')
    blend(p,np.ones(p.shape[1])/p.shape[1])
    counts=np.zeros(p.shape[1],int);total=np.zeros(len(p));history=[]
    best=np.inf;best_weights=None
    for t in range(1,steps+1):
        losses=[binary_loss(y,(total+p[:,j])/t) for j in range(p.shape[1])]
        chosen=int(np.flatnonzero(np.isclose(losses,min(losses),rtol=0,atol=1e-12))[0]);counts[chosen]+=1;total+=p[:,chosen]
        history.append(dict(step=t,chosen=chosen,chosen_loss=losses[chosen],losses=losses))
        if losses[chosen]<best-1e-12:
            best=losses[chosen];best_weights=counts.copy()/t
    return best_weights,history

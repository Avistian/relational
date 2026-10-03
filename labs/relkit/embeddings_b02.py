"""Visible B02 mechanisms. NumPy teaching operators, not a training replacement."""
import copy
import numpy as np

def piecewise_linear(x, edges):
    """One feature, train-fitted strictly increasing knots; extrapolate end bins."""
    x=np.asarray(x,dtype=float);edges=np.asarray(edges,dtype=float)
    if edges.ndim!=1 or len(edges)<3 or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):
        raise ValueError('At least three finite, strictly increasing edges required')
    if not np.isfinite(x).all():raise ValueError('Finite inputs required')
    z=(x[...,None]-edges[:-1])/np.diff(edges)
    result=np.clip(z,0,1)
    result[...,0]=np.minimum(z[...,0],1)
    result[...,-1]=np.maximum(z[...,-1],0)
    return result

def member_predictions(x, shared, r, s, bias):
    """One TabM-style affine layer: [batch,member,out]; no activation here."""
    x=np.asarray(x);shared=np.asarray(shared);r=np.asarray(r);s=np.asarray(s);bias=np.asarray(bias)
    if x.ndim!=2 or shared.ndim!=2 or r.ndim!=2 or s.ndim!=2 or bias.shape!=s.shape or r.shape[0]!=s.shape[0] or x.shape[1]!=r.shape[1] or shared.shape!=(r.shape[1],s.shape[1]):
        raise ValueError('Incompatible batch/member/feature dimensions')
    return np.einsum('bki,io->bko',x[:,None,:]*r[None,:,:],shared)*s[None,:,:]+bias[None,:,:]

def greedy_validation(predictions, target, max_size=32):
    """Fixed-pool regression exercise; allow repeats, first tie, strict improvement.

    This is not TabPack's online lifecycle or early-stopping implementation.
    Only validation arrays belong here; use selected indices unchanged on test.
    """
    p=np.asarray(predictions,dtype=float);y=np.asarray(target,dtype=float)
    if p.ndim!=2 or y.shape!=(p.shape[1],) or len(p)==0 or not np.isfinite(p).all() or not np.isfinite(y).all() or type(max_size)!=int or max_size<1:
        raise ValueError('Finite member-by-row predictions and positive size required')
    chosen=[];total=np.zeros_like(y);best=float('inf')
    for size in range(max_size):
        candidate=(total[None,:]+p)/(size+1)
        losses=np.mean((candidate-y)**2,axis=1)
        index=int(np.argmin(losses));loss=float(losses[index])
        if loss>=best:break
        chosen.append(index);total+=p[index];best=loss
    return chosen

def reserve_budget(state, phase, seconds, rate):
    """Immutable worst-case reservation including a 60-second startup allowance."""
    if not phase or phase in [r['phase'] for r in state['reservations']] or type(seconds)!=int or seconds<=0 or not np.isfinite(rate) or rate<=0:
        raise ValueError('Invalid or duplicate reservation')
    amount=(seconds+60)*rate
    total=state['overhead_usd']+sum(r['upper_usd'] for r in state['reservations'])+amount
    if total>min(state['cap_usd'],state['stop_usd']):raise ValueError('INCOMPLETE_BUDGET_GATE')
    result=copy.deepcopy(state)
    result['reservations'].append(dict(phase=phase,seconds=seconds,upper_usd=amount))
    return result

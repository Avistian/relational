"""Three live contracts for a defensible selected-experiment reproduction."""
import numpy as np

def verify_numeric_layout(columns, moments, checkpoint_mean, checkpoint_std):
    """Check ordered numerical feature semantics against saved encoder buffers."""
    expected=np.asarray([moments[c] for c in columns],dtype=float)
    mean=np.asarray(checkpoint_mean,dtype=float);std=np.asarray(checkpoint_std,dtype=float)
    if expected.shape!=(len(columns),2) or mean.shape!=(len(columns),) or std.shape!=mean.shape:
        raise ValueError('Numerical column count differs')
    if not np.isfinite(expected).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Nonfinite normalization evidence')
    if not np.allclose(expected[:,0],mean,rtol=1e-6,atol=1e-6) or not np.allclose(expected[:,1],std,rtol=1e-6,atol=1e-6):
        raise ValueError('Ordered feature moments differ')
    return True

def first_validation_min(history):
    """First minimum on a finite, contiguous validation history; never test scores."""
    if not history or [x['epoch'] for x in history]!=list(range(1,len(history)+1)):
        raise ValueError('History must be complete and ordered from epoch1')
    values=np.asarray([x['val_mae'] for x in history],dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite validation metric')
    return int(np.argmin(values))+1

def evidence_verdict(runs):
    """Frozen five-seed/ten-epoch scope. Closeness never identifies historical runs."""
    complete=(len(runs)==5 and sorted(x['seed'] for x in runs)==list(range(5)) and
        all(x['kind']=='RECONSTRUCTED_TRAINING' and x['epochs']==10 and x['complete'] and np.isfinite(x['test_mae']) for x in runs))
    if not complete:return dict(execution='INCOMPLETE',score='NOT_ESTABLISHED',historical='NOT_ESTABLISHED')
    mean=float(np.mean([x['test_mae'] for x in sorted(runs,key=lambda x:x['seed'])]))
    close=abs(mean-3.798)<=.20+1e-12
    return dict(execution='COMPLETE',score='CLOSE' if close else 'OUTSIDE_TOLERANCE',historical='NOT_ESTABLISHED')

"""Learner-owned checkpoint contracts; no model or training hidden here."""
import numpy as np

def select_checkpoint(history):
    """Return the first epoch with maximum finite validation AUROC."""
    if not history or [r['epoch'] for r in history] != list(range(1,len(history)+1)):
        raise ValueError('Require complete, ordered epochs starting at one')
    values=np.asarray([r['val']['roc_auc'] for r in history],dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Invalid validation AUROC')
    return history[int(np.argmax(values))]['epoch']

def align_predictions(query_keys, prediction_keys, predictions):
    """Return probabilities in query order; require identical unique key sets."""
    query_keys=list(map(tuple,query_keys));prediction_keys=list(map(tuple,prediction_keys))
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=1 or len(p)!=len(prediction_keys) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probability vector')
    if len(set(query_keys))!=len(query_keys) or len(set(prediction_keys))!=len(prediction_keys):
        raise ValueError('Duplicate entity/cutoff key')
    if set(query_keys)!=set(prediction_keys):raise ValueError('Prediction and query populations differ')
    lookup=dict(zip(prediction_keys,p))
    return np.asarray([lookup[key] for key in query_keys])

def reproduction_verdict(values, expected_seeds, target, tolerance, protocol_match):
    """Separate execution completeness, numerical agreement and protocol evidence."""
    expected=list(expected_seeds)
    if len(expected)<2 or len(set(expected))!=len(expected):raise ValueError('Require at least two unique seeds')
    if not set(values)<=set(expected):raise ValueError('Unexpected seed')
    a=np.asarray(list(values.values()),dtype=float)
    if not np.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('Invalid seed AUROC')
    if not np.isfinite(target) or not 0<=target<=1 or not np.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid comparison')
    complete=set(values)==set(expected)
    mean=float(np.mean(a)) if complete else None
    # Eight machine epsilons protect an inclusive decimal boundary; no metric-scale slack.
    rounding=8*np.finfo(float).eps
    return dict(execution='COMPLETE' if complete else 'INCOMPLETE',mean=mean,
                sample_sd=float(np.std(a,ddof=1)) if complete else None,
                score=('CLOSE' if abs(mean-target)<=tolerance+rounding else 'OUTSIDE_TOLERANCE') if complete else 'NOT_EVALUATED',
                protocol='ALIGNED_WITH_RELEASE' if protocol_match else 'GAPPED',
                historical_identity='NOT_ESTABLISHED')

"""Visible evidence contracts for Lesson 147; no training is performed."""
import math
import numpy as np

def keyed_losses(reference_keys, reference_targets, run_keys, run_targets, predictions):
    """Return absolute errors in reference order; never align by row position."""
    ref=[tuple(k) for k in reference_keys]; keys=[tuple(k) for k in run_keys]
    y=np.asarray(reference_targets,dtype=float); recorded=np.asarray(run_targets,dtype=float)
    pred=np.asarray(predictions,dtype=float)
    if not ref or len(set(ref))!=len(ref) or len(set(keys))!=len(keys):
        raise ValueError('Empty reference or duplicate query keys')
    if set(ref)!=set(keys) or any(len(k)!=2 for k in ref+keys):
        raise ValueError('Require identical complete (entity, cutoff) sets')
    if y.shape!=(len(ref),) or recorded.shape!=(len(keys),) or pred.shape!=(len(keys),):
        raise ValueError('Require one scalar target/prediction per key')
    if not all(np.isfinite(a).all() for a in [y,recorded,pred]):
        raise ValueError('Nonfinite target or prediction')
    index={key:i for i,key in enumerate(keys)}
    order=[index[key] for key in ref]
    if not np.array_equal(recorded[order],y):
        raise ValueError('Recorded targets disagree with reference targets')
    return np.abs(pred[order]-y)

def paired_summary(left, right):
    """Summarize left-minus-right MAE across matched seeds, with sample SD."""
    if set(left)!=set(right) or len(left)<2:
        raise ValueError('Need identical sets of at least two seeds')
    seeds=sorted(left)
    differences=np.array([left[s]-right[s] for s in seeds],dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError('Nonfinite paired scores')
    return dict(seeds=seeds,differences=differences.tolist(),mean=float(differences.mean()),
                sample_sd=float(differences.std(ddof=1)))

def rank_questions(rows, max_hours, max_usd):
    """Planning rule: feasible first; stronger evidence, fewer hours, stable ID.

    Hours/cost/readiness/evidence are declared judgments, not fitted quantities.
    Each question is considered independently, not as an additive portfolio.
    """
    if not all(math.isfinite(x) and x>=0 for x in [max_hours,max_usd]):
        raise ValueError('Invalid planning limits')
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate question ID')
    for r in rows:
        if not all(math.isfinite(r[k]) and r[k]>=0 for k in ['hours','usd']):
            raise ValueError('Invalid planning cost')
        if r['evidence'] not in [0,1,2,3] or type(r['ready']) is not bool:
            raise ValueError('Invalid evidence level or readiness')
    feasible=[r for r in rows if r['ready'] and r['hours']<=max_hours and r['usd']<=max_usd]
    return [r['id'] for r in sorted(feasible,key=lambda r:(-r['evidence'],r['hours'],r['id']))]

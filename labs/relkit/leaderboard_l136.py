"""Visible leaderboard audit primitives; standard-library only."""
import math

def align_predictions(expected_keys, prediction_keys, predictions):
    """Return predictions in official query order after checking a bijection."""
    expected=[tuple(k) for k in expected_keys]
    supplied=[tuple(k) for k in prediction_keys]
    values=[float(v) for v in predictions]
    if not expected or len(supplied)!=len(values):
        raise ValueError('Empty queries or different key/value counts')
    for keys in [expected,supplied]:
        if any(not k or any(x is None or (isinstance(x,float) and not math.isfinite(x)) for x in k) for k in keys):
            raise ValueError('Missing query key')
        if len(set(keys))!=len(keys):
            raise ValueError('Duplicate query key; entity alone is not a temporal query')
    if set(expected)!=set(supplied):
        raise ValueError('Missing or foreign prediction keys')
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite prediction')
    lookup=dict(zip(supplied,values))
    return [lookup[k] for k in expected]

def regression_score(targets, predictions, train_std):
    """NMAE uses the pinned train-target sample SD, never a test-fitted scale."""
    y=[float(v) for v in targets];p=[float(v) for v in predictions];scale=float(train_std)
    if not y or len(y)!=len(p) or not math.isfinite(scale) or scale<=0:
        raise ValueError('Need paired rows and a finite positive train scale')
    if not all(math.isfinite(v) for v in y+p):
        raise ValueError('Nonfinite target or prediction')
    mae=math.fsum(abs(a-b) for a,b in zip(y,p))/len(y)
    return dict(mae=mae,nmae=mae/scale,count=len(y))

def complete_board(scores, canonical_tasks):
    """Equal task weighting is allowed only for exactly the canonical task set."""
    tasks=list(canonical_tasks)
    if not tasks or len(set(tasks))!=len(tasks) or set(scores)!=set(tasks):
        raise ValueError('Incomplete, extra or duplicate board tasks')
    values=[float(scores[t]) for t in tasks]
    if not all(math.isfinite(v) and v>=0 for v in values):
        raise ValueError('Invalid regression score')
    return math.fsum(values)/len(values)

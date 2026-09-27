"""Visible learner-owned feature, identity and selection contracts for Lesson 129."""
import math

def past_summary(events, entity, cutoff, lookback):
    if not math.isfinite(lookback) or lookback <= 0:
        raise ValueError('lookback must be positive')
    eligible = [r for r in events if r['entity'] == entity
                and cutoff - lookback < r['event'] < cutoff
                and r['available'] <= cutoff]
    if not eligible:
        return dict(count=0, mean=None, days_since_latest=None)
    return dict(count=len(eligible),
                mean=math.fsum(r['value'] for r in eligible) / len(eligible),
                days_since_latest=cutoff - max(r['event'] for r in eligible))

def align_predictions(feature_keys, predictions, query_keys):
    feature_keys = [tuple(k) for k in feature_keys]
    query_keys = [tuple(k) for k in query_keys]
    if len(feature_keys) != len(predictions):
        raise ValueError('one prediction per feature row required')
    if len(set(feature_keys)) != len(feature_keys) or len(set(query_keys)) != len(query_keys):
        raise ValueError('duplicate query keys')
    if set(feature_keys) != set(query_keys):
        raise ValueError('query key sets differ')
    if not all(math.isfinite(float(p)) for p in predictions):
        raise ValueError('nonfinite prediction')
    mapping = dict(zip(feature_keys, predictions))
    return [float(mapping[k]) for k in query_keys]

def choose_trial(trials):
    if not trials or len({r['number'] for r in trials}) != len(trials):
        raise ValueError('need complete unique trials')
    if not all(math.isfinite(r['val_mae']) and r['val_mae'] >= 0 for r in trials):
        raise ValueError('invalid validation MAE')
    return min(trials, key=lambda r: (r['val_mae'], r['number']))['number']

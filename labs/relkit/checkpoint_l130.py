"""Learner-owned identity and evidence checks for the Q1 RelBench checkpoint."""
import math
import statistics

def validate_queries(rows, expected_count):
    """Validate unique (entity, time) queries with finite labels; return their keys."""
    if expected_count < 1 or len(rows) != expected_count:
        raise ValueError('Incomplete or empty query set')
    keys = []
    try:
        for row in rows:
            entity, time = row['entity'], row['time']
            if (isinstance(entity, bool) or isinstance(time, bool)
                    or int(entity) != entity or int(time) != time
                    or not math.isfinite(float(row['target']))):
                raise ValueError('Invalid query key or target')
            keys.append((int(entity), int(time)))
        if len(set(keys)) != expected_count:
            raise ValueError('Duplicate query key')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Malformed query') from exc
    return keys

def keyed_mae(queries, predictions):
    """Align predictions to exact query keys and return MAE; reject partial coverage."""
    keys = validate_queries(queries, len(queries))
    try:
        prediction_rows = [dict(entity=p['entity'], time=p['time'], target=p['prediction'])
                           for p in predictions]
    except (KeyError, TypeError) as exc:
        raise ValueError('Malformed prediction') from exc
    prediction_keys = validate_queries(prediction_rows, len(queries))
    if set(prediction_keys) != set(keys):
        raise ValueError('Prediction keys differ from task keys')
    lookup = dict(zip(prediction_keys, (float(p['target']) for p in prediction_rows)))
    return math.fsum(abs(lookup[key] - float(q['target']))
                     for key, q in zip(keys, queries)) / len(keys)

def reproduction_verdict(records, expected_seeds=(0,1,2,3,4), tolerance=.2):
    """Verify complete distinct fits before comparing mean MAEs to Table7 targets."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid tolerance')
    try:
        if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
            raise ValueError('Need distinct expected seeds')
        if len(records) != len(expected_seeds) or {r['seed'] for r in records} != set(expected_seeds):
            raise ValueError('Missing, duplicate or unexpected seeds')
        if len({r['run_uuid'] for r in records}) != len(records):
            raise ValueError('Reused run identity')
        for r in records:
            if r['status'] != 'COMPLETE' or r['epochs'] != 10 or not r['run_uuid']:
                raise ValueError('Incomplete run')
            if any(not math.isfinite(r[s]) or r[s] < 0 for s in ['val', 'test']):
                raise ValueError('Invalid MAE')
        output = {}
        for split, target in [('val', 3.193), ('test', 4.022)]:
            values = [r[split] for r in sorted(records, key=lambda r:r['seed'])]
            mean = statistics.mean(values)
            output[split] = dict(mean=mean, sample_sd=statistics.stdev(values), n=len(values),
                target=target, delta=mean-target, descriptive_tolerance=tolerance,
                verdict='CLOSE' if abs(mean-target) <= tolerance else 'OUTSIDE_TOLERANCE')
        return output
    except (KeyError, TypeError) as exc:
        raise ValueError('Malformed run evidence') from exc

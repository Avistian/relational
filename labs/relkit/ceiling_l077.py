"""L077: complete finite-support experiment; no external training data or model."""
import hashlib
import json
import numpy as np


def make_database(n_pairs=1000, seed=0):
    """Opposite histories share every flat feature. IDs are metadata, never inputs."""
    if n_pairs < 10:
        raise ValueError('Use at least ten pairs for nonempty partitions')
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(10001, 10001 + 2*n_pairs))
    customers, events = [], []
    for pair in range(n_pairs):
        base, step = int(rng.integers(10, 101)), int(rng.integers(1, 21))
        own = int(rng.integers(0, 10))
        for target in (0, 1):
            cid = int(ids[2*pair + target])
            customers.append(dict(customer_id=cid, pair_id=pair, own=own, target=target))
            amounts = [base, base+step, base+2*step]
            if target == 0:
                amounts.reverse()
            for day, amount in enumerate(amounts, 1):
                events.append(dict(customer_id=cid,event_day=day,available_day=day,amount=amount))
            # Both must be invisible at cutoff10, including the old but late row.
            events.append(dict(customer_id=cid,event_day=12,available_day=12,amount=10000))
            events.append(dict(customer_id=cid,event_day=2,available_day=11,amount=-10000))
    rng.shuffle(customers)
    rng.shuffle(events)
    return customers, events


def split_pairs(customers, seed=0):
    """60/20/20 disjoint pair groups. Test is untouched during fitting."""
    groups = np.unique([c['pair_id'] for c in customers])
    groups = np.random.default_rng(seed+10000).permutation(groups)
    a, b = int(.6*len(groups)), int(.8*len(groups))
    return tuple(np.array([i for i,c in enumerate(customers) if c['pair_id'] in set(part)],dtype=int)
                 for part in (groups[:a],groups[a:b],groups[b:]))


def eligible_histories(customers, events, cutoff):
    """Map identity to eligible rows; protect both event time and availability."""
    groups = {c['customer_id']: [] for c in customers}
    if len(groups) != len(customers):
        raise ValueError('Customer primary keys must be unique')
    for e in events:
        if e['customer_id'] not in groups:
            raise ValueError('Unknown customer foreign key')
        if e['event_day'] <= cutoff and e['available_day'] <= cutoff:
            groups[e['customer_id']].append(e)
    return groups


def flat_rows(customers, events, cutoff):
    """[own, count, sum, mean, max]; empty history aggregates are zero."""
    histories = eligible_histories(customers, events, cutoff)
    rows = []
    for c in customers:
        a = [e['amount'] for e in histories[c['customer_id']]]
        n = len(a)
        rows.append([c['own'],n,sum(a),sum(a)/n if n else 0,max(a) if n else 0])
    return np.asarray(rows,dtype=float)


def temporal_delta(customers, events, cutoff):
    """Last minus first eligible amount, using event time; empty/singleton → 0."""
    histories = eligible_histories(customers, events, cutoff)
    result = []
    for c in customers:
        h = sorted(histories[c['customer_id']],key=lambda e:e['event_day'])
        if len({e['event_day'] for e in h}) != len(h):
            raise ValueError('Tied eligible event times need an explicit ordering rule')
        result.append(h[-1]['amount']-h[0]['amount'] if h else 0)
    return np.asarray(result,dtype=float)


def collision_ceiling(x, y):
    """Exact empirical max accuracy of any deterministic function of x alone."""
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or len(x) != len(y) or not len(y):
        raise ValueError('Need a nonempty row matrix and aligned labels')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features have no declared equality semantics')
    groups = {}
    for row, label in zip(x,y):
        counts = groups.setdefault(tuple(row),{})
        counts[int(label)] = counts.get(int(label),0)+1
    return sum(max(counts.values()) for counts in groups.values())/len(y)


def fit_stump(x, y):
    """Train a depth-one classifier by exhaustive threshold search; ties stay first."""
    x, y = np.asarray(x), np.asarray(y)
    best, best_correct = None, -1
    for col in range(x.shape[1]):
        values = np.unique(x[:,col])
        thresholds = np.r_[values[0]-1,(values[:-1]+values[1:])/2,values[-1]+1]
        for threshold in thresholds:
            for right_label in (0,1):
                pred = np.where(x[:,col]>threshold,right_label,1-right_label)
                correct = int(np.sum(pred==y))
                if correct > best_correct:
                    best_correct = correct
                    best = dict(column=col,threshold=float(threshold),right_label=right_label)
    return best


def predict_stump(model, x):
    """Prediction consumes only the supplied row; no label/ID lookup."""
    return np.where(np.asarray(x)[:,model['column']]>model['threshold'],
                    model['right_label'],1-model['right_label'])


def run_experiment(n_pairs=1000, seed=0):
    """One full protocol run; all representations use the student's live helpers."""
    customers, events = make_database(n_pairs,seed)
    train, valid, test = split_pairs(customers,seed)
    y = np.array([c['target'] for c in customers])
    flat = flat_rows(customers,events,10)
    restored = np.column_stack([flat,temporal_delta(customers,events,10)])
    scores, models, ceilings = {}, {}, {}
    for name, x in [('flat',flat),('restored',restored)]:
        model = fit_stump(x[train],y[train])
        models[name] = model
        scores[name] = {part:float(np.mean(predict_stump(model,x[idx])==y[idx]))
                       for part,idx in [('train',train),('validation',valid),('test',test)]}
        # Audit uses test labels AFTER model freeze; never a training input.
        ceilings[name] = collision_ceiling(x[test],y[test])
    digest = hashlib.sha256(json.dumps([customers,events],sort_keys=True).encode()).hexdigest()
    return dict(seed=seed,split_seed=seed+10000,n_pairs=n_pairs,cutoff=10,
                split_rows=[len(train),len(valid),len(test)],dataset_sha256=digest,
                scores=scores,models=models,ceilings=ceilings)

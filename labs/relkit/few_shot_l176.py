"""Visible learner contracts: nested supports, keyed ranking and paired context gains."""
def nested_support(n_rows, contexts, seed_key):
    import hashlib
    import numpy as np
    if type(n_rows) is not int or not contexts or any(type(k) is not int or not 0<k<=n_rows for k in contexts) or len(set(contexts))!=len(contexts) or not isinstance(seed_key,str) or not seed_key:
        raise ValueError('Invalid support request')
    seed=int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4],'big')
    largest=np.random.default_rng(seed).choice(n_rows,max(contexts),replace=False)
    return {k:largest[:k].copy() for k in contexts}


def keyed_auc(label_keys, labels, prediction_keys, probabilities):
    import numpy as np
    lk=np.asarray(label_keys);pk=np.asarray(prediction_keys);y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if lk.ndim!=2 or pk.ndim!=2 or lk.shape[1]!=2 or pk.shape[1]!=2 or y.shape!=(len(lk),) or p.shape!=(len(pk),):raise ValueError('Invalid keyed arrays')
    left=list(map(tuple,lk));right=list(map(tuple,pk))
    if len(set(left))!=len(left) or len(set(right))!=len(right) or set(left)!=set(right):raise ValueError('Missing or duplicate composite keys')
    if not np.isfinite(p).all() or not ((0<=p)&(p<=1)).all() or set(y.tolist())!={0,1}:raise ValueError('Invalid labels or probabilities')
    lookup=dict(zip(right,p));aligned=np.array([lookup[k] for k in left])
    positive=aligned[y==1,None];negative=aligned[None,y==0]
    return float(np.mean((positive>negative)+.5*(positive==negative)))


def paired_curve(records, contexts, seeds):
    import numpy as np
    if not contexts or len(set(contexts))!=len(contexts) or contexts!=sorted(contexts) or len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('Invalid curve specification')
    expected={(k,s) for k in contexts for s in seeds};values={}
    for r in records:
        key=(r['context'],r['seed']);value=float(r['auc'])
        if key not in expected or key in values or not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid curve point')
        values[key]=value
    if set(values)!=expected:raise ValueError('Incomplete curve')
    levels=[];doublings=[]
    for i,k in enumerate(contexts):
        a=np.array([values[k,s] for s in seeds])
        levels.append(dict(context=k,per_seed=a.tolist(),mean=float(a.mean()),sample_sd=float(a.std(ddof=1))))
        if i:
            delta=a-np.array([values[contexts[i-1],s] for s in seeds])
            doublings.append(dict(start=contexts[i-1],end=k,per_seed=delta.tolist(),mean_gain=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),positive_seeds=int((delta>0).sum())))
    return dict(levels=levels,doublings=doublings,interpretation='SUPPORT_DRAW_VARIATION_ON_ONE_FIXED_TASK')


def reserve_cost(budget, attempt, seconds, rate):
    import copy, math
    if not isinstance(attempt,str) or not attempt or not math.isfinite(seconds) or seconds<=0 or not math.isfinite(rate) or rate<=0:raise ValueError('Invalid reservation')
    b=copy.deepcopy(budget)
    if any(r['attempt']==attempt for r in b['reservations']):raise ValueError('Immutable attempt already reserved')
    r=dict(attempt=attempt,seconds=seconds,lifecycle_seconds=30,upper_usd=(seconds+30)*rate)
    total=b['overhead_usd']+sum(x['upper_usd'] for x in b['reservations'])+r['upper_usd']
    if total>min(b['cap_usd'],b['planned_stop_usd']):raise ValueError('INCOMPLETE_BUDGET_GATE')
    b['reservations'].append(r);b['reserved_usd']=total
    return b

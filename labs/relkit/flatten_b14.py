"""Visible B14 course mechanisms. Ridge/RBF are diagnostics, not TabPFN models.

Event records: (event_id, entity_id, event_time, available_time, value).
Query identity: (entity_id, prediction_time). All times are integer days.
"""
import hashlib,json
import numpy as np


def flatten(entities, events, queries, snapshot):
    """One row/query: local value, count, mean, last value. No label input.

    Event time is strictly before min(query time, database snapshot).
    Arrival time may equal that boundary. Duplicate events are rejected rather
    than multiplied by a join. Empty histories have count/mean/last all zero.
    """
    if len({e[0] for e in events}) != len(events):
        raise ValueError('Duplicate physical event')
    if len(set(map(tuple,queries))) != len(queries):
        raise ValueError('Duplicate query identity')
    rows=[]
    for entity, time in queries:
        if entity not in entities: raise ValueError('Unknown entity')
        cutoff=min(time,snapshot)
        visible=[e for e in events if e[1]==entity and e[2]<cutoff and e[3]<=cutoff]
        visible.sort(key=lambda e:(e[2],e[0]))
        values=[e[4] for e in visible]
        rows.append([entities[entity],len(values),sum(values)/len(values) if values else 0.,values[-1] if values else 0.])
    return np.asarray(rows,dtype=np.float64)


def standardize(support, query):
    """Freeze mean/population SD from the support, including constant columns."""
    mean=np.mean(support,axis=0);scale=np.std(support,axis=0)
    scale=np.where(scale==0,1.,scale)
    return (support-mean)/scale,(query-mean)/scale


def predict(support, labels, query, backbone):
    """Fixed alpha=1. Mean-centered y; explicit linear or RBF kernel solve.

    The inputs must already be standardized from support only. The mean target
    is the unpenalized offset. RBF gamma=1/feature_count. No hyperparameter tuning.
    """
    support=np.asarray(support);query=np.asarray(query);labels=np.asarray(labels)
    offset=float(labels.mean());y=labels-offset
    if backbone=='ridge':
        coef=np.linalg.solve(support.T@support+np.eye(support.shape[1]),support.T@y)
        return offset+query@coef
    if backbone=='rbf':
        def kernel(a,b):
            dist=np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
            return np.exp(-dist/support.shape[1])
        weights=np.linalg.solve(kernel(support,support)+np.eye(len(support)),y)
        return offset+kernel(query,support)@weights
    raise ValueError('Unknown backbone')


def keyed_mse(keys, labels, predictions):
    """Require an exact bijection of full keys before scoring."""
    keys=list(map(tuple,keys));got=[tuple(r['key']) for r in predictions]
    if len(set(keys))!=len(keys) or len(set(got))!=len(got) or set(keys)!=set(got):
        raise ValueError('Prediction keys are not a complete bijection')
    lookup={tuple(r['key']):float(r['prediction']) for r in predictions}
    a=np.array([lookup[k] for k in keys]);y=np.asarray(labels)
    if len(y)!=len(keys) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(y)):
        raise ValueError('Invalid prediction/label values')
    return float(np.mean((a-y)**2))


def make_world(seed):
    """128 entities; an independent local feature and a nonlinear history signal."""
    rng=np.random.default_rng(seed)
    entities={i:float(v) for i,v in enumerate(rng.normal(size=128))}
    latent=rng.uniform(-3,3,128);events=[]
    for day in range(0,120,5):
        for i in entities:
            value=float(latent[i]+.6*np.sin(day/12+i)+rng.normal(scale=.2))
            events.append((len(events),i,day,day+1,value))
    support=[(i,t) for t in [20,40,60,80] for i in entities]
    query=[(i,t) for t in [100,110] for i in entities]
    # Outcomes depend on history up to each query, representing a rolling world.
    # Evaluation features are frozen at day100, so later queries face staleness.
    def labels(keys):
        f=flatten(entities,events,keys,120)
        return .4*f[:,0]+1.5*np.sin(f[:,2])+.15*f[:,3]+rng.normal(scale=.1,size=len(keys))
    return dict(entities=entities,events=events,support=support,query=query,y=labels(support),target=labels(query))


def run_diagnostic():
    """12 fits, two representations x two predictors x three paired seeds.

    Final support includes validation day80: labels mature by day90. No tuning
    uses those labels. Rolling predictions reuse fitted support and change only
    query features; they are a different information policy, not a new fit.
    """
    conditions=[]
    for seed in range(3):
        w=make_world(seed);s=flatten(w['entities'],w['events'],w['support'],100)
        q=flatten(w['entities'],w['events'],w['query'],100)
        rolling=flatten(w['entities'],w['events'],w['query'],120)
        extra=w['events']+[(999999,0,105,106,1e6)]
        assert np.array_equal(q,flatten(w['entities'],extra,w['query'],100))
        for feature,cols in [('entity-only',[0]),('relational',[0,1,2,3])]:
            a,b=standardize(s[:,cols],q[:,cols]);_,c=standardize(s[:,cols],rolling[:,cols])
            identity=hashlib.sha256(a.tobytes()+b.tobytes()+json.dumps(w['support']).encode()).hexdigest()
            for backbone in ['ridge','rbf']:
                both=predict(a,w['y'],np.vstack([b,c]),backbone);scores=both[:len(b)];roll=both[len(b):]
                rows=[dict(key=list(k),label=float(y),prediction=float(p),rolling_prediction=float(r)) for k,y,p,r in zip(w['query'],w['target'],scores,roll)]
                conditions.append(dict(seed=seed,features=feature,backbone=backbone,input_sha256=identity,support_rows=len(a),query_rows=len(b),mse=keyed_mse(w['query'],w['target'],rows),rolling_mse=float(np.mean((roll-w['target'])**2)),predictions=rows))
    return dict(status='COMPLETE_COURSE_DIAGNOSTIC',fits=12,predictions=3072,rolling_predictions=3072,seed_count=3,scope='Synthetic regression; not a foundation-model or paper reproduction',conditions=conditions)

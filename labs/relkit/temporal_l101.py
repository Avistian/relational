"""L101: visible temporal boundaries and full RelBench heuristic replay."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TASK_URL='https://relbench.stanford.edu/download/rel-f1/tasks/driver-position.zip'
TASK_SHA256='775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e'
ARMS=['global_zero','global_mean','global_median','entity_mean','entity_median']
TARGETS={'val':[11.083,4.334,4.136,7.181,7.114], 'test':[11.926,4.513,4.399,8.501,8.519]}

def available(event_time, available_time, query_time):
    """Inclusive observation convention: both occurrence and arrival must be known."""
    return bool(event_time <= query_time and available_time <= query_time)

def split_queries(queries, val_time, test_time):
    """Purge labels that mature beyond the next fit/evaluation boundary."""
    if val_time >= test_time:
        raise ValueError('Validation must precede test')
    out=[]
    for q in queries:
        t,r=q['time'],q['label_ready']
        if r<t:
            raise ValueError('A future-window label cannot mature before its query')
        if t<val_time:
            out.append('train' if r<=val_time else 'purged')
        elif t<test_time:
            out.append('val' if r<=test_time else 'purged')
        else:
            out.append('test')
    return out

def incoming_subgraph(nodes, edges, seed, query_time, depth):
    """All-neighbor incoming expansion; keep the SAME query cutoff at every hop.

    nodes maps typed IDs to (event_time, available_time).
    edges stores (source, destination, edge_available_time).
    The returned edge indices refer to the original edge list, including duplicates.
    This is a visibility audit, not a trained GNN or a temporal-walk model.
    """
    if depth<0 or seed not in nodes:
        raise ValueError('Invalid seed or depth')
    if not available(*nodes[seed],query_time):
        raise ValueError('Seed did not exist at query time')
    frontier={seed};seen={seed};used=set()
    for _ in range(depth):
        following=set()
        for i,(src,dst,edge_ready) in enumerate(edges):
            if dst in frontier and edge_ready<=query_time and available(*nodes[src],query_time) and available(*nodes[dst],query_time):
                following.add(src);used.add(i)
        seen |= following;frontier=following
    return seen,used

def load_task(cache_dir):
    """Read all three released tables after validating the release archive hash."""
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/'driver-position.zip'
    if not path.exists():
        with urllib.request.urlopen(TASK_URL,timeout=60) as response:
            raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=TASK_SHA256:
            raise ValueError('Downloaded task bytes differ from v1.1.0')
        path.write_bytes(raw)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TASK_SHA256:
        raise ValueError('Cached task bytes differ from v1.1.0')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return {s:pd.read_parquet(io.BytesIO(archive.read(f'driver-position/{s}.parquet'))) for s in ['train','val','test']}

def baseline_predict(fit_table, query_table, arm):
    """Fit only on permitted labels. Query tables need only driverId and date.

    Entity baselines use zero for an unseen driver, matching the pinned release.
    No optimizer, stochastic initialization, feature preprocessing or tuning occurs.
    """
    if arm=='global_zero':
        return np.zeros(len(query_table))
    if arm=='global_mean':
        return np.full(len(query_table),fit_table.position.mean())
    if arm=='global_median':
        return np.full(len(query_table),fit_table.position.median())
    if arm in ['entity_mean','entity_median']:
        values=fit_table.groupby('driverId').position.agg(arm.split('_')[1])
        return query_table.driverId.map(values).fillna(0).to_numpy(dtype=float)
    raise ValueError(arm)

def paper_replay(cache_dir, prediction_path=None):
    """Full five-baseline, two-split published target; fixed rounding tolerance .0005."""
    tables=load_task(cache_dir)
    val_time=pd.Timestamp('2005-01-01');test_time=pd.Timestamp('2010-01-01')
    horizon=pd.Timedelta(days=60)
    assert (tables['train'].date+horizon<=val_time).all()
    assert (tables['val'].date>=val_time).all()
    assert (tables['val'].date+horizon<=test_time).all()
    assert (tables['test'].date>=test_time).all()
    records=[];predictions={}
    for split in ['val','test']:
        # The released baseline_node.py explicitly refits on train+val for test.
        fit=tables['train'] if split=='val' else pd.concat([tables['train'],tables['val']],ignore_index=True)
        queries=tables[split][['driverId','date']].copy()
        for i,arm in enumerate(ARMS):
            pred=baseline_predict(fit,queries,arm)
            truth=tables[split].position.to_numpy()
            mae=float(np.abs(pred-truth).mean())
            target=TARGETS[split][i]
            records.append({'split':split,'arm':arm,'fit_rows':len(fit),'evaluation_rows':len(queries),'mae':mae,'paper_mae':target,'absolute_gap':abs(mae-target),'verdict':'MATCH' if abs(mae-target)<=.0005 else 'FAIL'})
            predictions[f'{split}_{arm}']=pred
        predictions[f'{split}_target']=tables[split].position.to_numpy()
        predictions[f'{split}_driverId']=queries.driverId.to_numpy()
        predictions[f'{split}_date_ns']=queries.date.to_numpy(dtype='datetime64[ns]').astype('int64')
    if prediction_path:
        np.savez_compressed(prediction_path,**predictions)
    return {'status':'COMPLETE','target':'Robinson et al. arXiv:2407.20060v1 Table 4, rel-f1/driver-position, all five heuristic columns',
            'source_commit':'9aa346267c2e1c560bd92da07d6f4ad1ca2f0639','data_sha256':TASK_SHA256,
            'counts':{s:len(t) for s,t in tables.items()},'seed_policy':'deterministic; repetitions would duplicate predictions',
            'tolerance':.0005,'results':records,'historical_execution_identity':'NOT_ESTABLISHED','full_paper_reproduction':'NOT_ESTABLISHED',
            'unrun':['LightGBM and RDL columns','other tasks and databases','reconstruction of task labels from original database']}

def course_experiment():
    """Nine fresh fits: a deliberately constructed future-information witness.

    Each query has one known zero-valued past record, a late-arriving record
    carrying its label, and a future outcome record carrying the same label.
    Independent Bernoulli labels ensure there is no legitimate past signal.
    This is a diagnostic construction, not an estimate of typical leakage uplift.
    """
    records=[];split_audit=[]
    for seed in [0,1,2]:
        rng=np.random.default_rng(seed);n=600
        times=np.arange(n);y=rng.integers(0,2,n)
        queries=[{'time':int(t),'label_ready':int(t+2)} for t in times]
        tags=np.array(split_queries(queries,360,480))
        train=np.where(tags=='train')[0];val=np.where(tags=='val')[0];test=np.where(tags=='test')[0]
        # Audit a random partition separately. It changes evaluation samples.
        perm=rng.permutation(n);random_train=perm[:360];random_test=perm[480:]
        bad_pairs=sum(queries[i]['label_ready']>queries[j]['time'] for i in random_train for j in random_test)
        split_audit.append({'seed':seed,'random_future_label_pairs':int(bad_pairs),'random_pairs':len(random_train)*len(random_test),'chronological_future_label_pairs':0,'purged':int(sum(tags=='purged'))})
        features={arm:[] for arm in ['legal','event_only','static']}
        for t,label in zip(times,y):
            events=[(int(t-1),int(t-1),0.),(int(t-1),int(t+1),float(label)),(int(t+1),int(t+1),float(label))]
            features['legal'].append(np.mean([v for e,a,v in events if available(e,a,int(t))]))
            features['event_only'].append(np.mean([v for e,a,v in events if e<=t]))
            features['static'].append(np.mean([v for e,a,v in events]))
        for arm,values in features.items():
            x=np.asarray(values).reshape(-1,1)
            model=LogisticRegression(C=1,solver='lbfgs',max_iter=200,random_state=seed).fit(x[train],y[train])
            records.append({'seed':seed,'arm':arm,'train':len(train),'val':len(val),'test':len(test),
                            'val_accuracy':float(model.score(x[val],y[val])),'test_accuracy':float(model.score(x[test],y[test])),
                            'coefficient':float(model.coef_[0,0]),'intercept':float(model.intercept_[0]),
                            'test_predictions':model.predict(x[test]).tolist(),'test_labels':y[test].tolist()})
    return {'status':'COMPLETE','fits':9,'records':records,'split_audit':split_audit,'paper_comparison':'INCOMPARABLE','claim':'Constructed leakage witness; not a real-world performance estimate'}

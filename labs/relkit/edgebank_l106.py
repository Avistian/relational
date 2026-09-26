"""Visible EdgeBank and evaluator. Released batch replay is distinct from strict causal fixtures."""
# %% Imports
from pathlib import Path
import hashlib, urllib.request, random
import numpy as np
DATA_URL='https://snap.stanford.edu/jodie/wikipedia.csv'
DATA_SHA='a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09'

# %% TODO 1: before-event history

def legal_history(times, availability, query):
    """Strict event clock; inclusive availability clock. Teaching contract."""
    t,a=np.asarray(times),np.asarray(availability)
    if t.ndim!=1 or a.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(a).all() or np.any(a<t) or not np.isfinite(query):
        raise ValueError('Aligned finite clocks, availability >= event time required')
    return (t<query)&(a<=query)

# %% TODO 2: memory membership

def memory_scores(history, candidates, mode='unlimited'):
    """history [N,3]=(source,destination,time); candidates [Q,2]; scores [Q].

    window matches released fixed mode: latest 15% quantile of supplied
    HISTORY EVENTS, not 15% of wall-clock time or a fixed-duration window.
    Caller establishes which history is legal. No current candidates inserted.
    """
    h,c=np.asarray(history),np.asarray(candidates)
    if mode not in ('unlimited','window'):raise ValueError('Unknown memory mode')
    if h.ndim!=2 or h.shape[1]!=3 or c.ndim!=2 or c.shape[1]!=2:raise ValueError('Expected [N,3] and [Q,2]')
    if not np.isfinite(h).all() or not np.isfinite(c).all():raise ValueError('Finite inputs required')
    if len(h) and mode=='window':h=h[h[:,2]>=np.quantile(h[:,2],.85)]
    memory=set(map(tuple,h[:,:2]))
    return np.array([tuple(pair) in memory for pair in c],dtype=np.int8)

# %% TODO 3: binary-score metrics with ties

def binary_metrics(positive, negative):
    """AP and AUROC for {0,1} scores, grouping ties at one threshold."""
    p,n=np.asarray(positive),np.asarray(negative)
    if p.ndim!=1 or n.ndim!=1 or not len(p) or not len(n) or not np.isin(np.r_[p,n],[0,1]).all():raise ValueError('Nonempty binary score vectors required')
    tp,fp=float(p.sum()),float(n.sum());P,N=len(p),len(n)
    high_precision=tp/(tp+fp) if tp+fp else 0.
    ap=(tp/P)*high_precision+(1-tp/P)*P/(P+N)
    auc=(tp*(N-fp)+.5*(tp*fp+(P-tp)*(N-fp)))/(P*N)
    return np.array([ap,auc])

# %% PROVIDED: authenticate and project the complete raw file

def load_events(path):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.download');urllib.request.urlretrieve(DATA_URL,tmp);tmp.replace(path)
    digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    if digest!=DATA_SHA:raise ValueError('Raw data hash mismatch')
    rows=[]
    with path.open() as f:
        next(f)
        for line in f:
            u,v,t,*_=line.split(',',3);rows.append((int(u)+1,int(v)+8228,float(t)))
    events=np.asarray(rows,dtype=np.float64)
    if events.shape!=(157474,3) or np.any(np.diff(events[:,2])<0):raise ValueError('Unexpected event population')
    return events

# %% PROVIDED: released split, explicit Python compatibility adaptation

def released_split(events):
    u,v,t=events.T;val_time,test_time=np.quantile(t,[.70,.85])
    nodes=set(u.astype(np.int64))|set(v.astype(np.int64))
    test_nodes=set(u[t>val_time].astype(np.int64))|set(v[t>val_time].astype(np.int64))
    # Python 3.9 random.sample(set) internally converted the set to tuple.
    held=set(random.Random(2020).sample(tuple(test_nodes),int(.1*len(nodes))))
    observed=np.array([a not in held and b not in held for a,b in zip(u,v)])
    train=(t<=val_time)&observed;valid=(t>val_time)&(t<=test_time);test=t>test_time
    return np.flatnonzero(train|valid),np.flatnonzero(test),sorted(map(int,held)),[float(val_time),float(test_time)]

# %% PROVIDED: complete replay on frozen original candidate arrays

def replay(events,history_ids,test_ids,negative_edges,batch_size=200):
    """Score before adding each batch; arithmetic-mean batch AP/AUC like source.

    negative_edges: [runs, test events, 2]. Also return pooled diagnostics,
    collisions and tie exposure counts; never confuse them with paper metrics.
    """
    runs=[];predictions=[]
    for neg in negative_edges:
        per_mode={k:[] for k in ('unlimited','window')};pred={k:[] for k in per_mode}
        collisions=ties=0
        for start in range(0,len(test_ids),batch_size):
            ids=test_ids[start:start+batch_size];h=events[np.r_[history_ids,test_ids[:start]]];pos=events[ids,:2];n=neg[start:start+len(ids)]
            collisions+=sum(tuple(x) in set(map(tuple,pos)) for x in n)
            ties+=int(np.sum(h[:,2]>=events[ids[0],2]))
            for mode in per_mode:
                scores=memory_scores(h,np.r_[pos,n],mode);p,q=np.split(scores,2)
                per_mode[mode].append(binary_metrics(p,q));pred[mode].append(np.stack([p,q],axis=1))
        r={}
        for mode in per_mode:
            pp=np.concatenate(pred[mode]);r[mode]={'batch_mean':np.mean(per_mode[mode],axis=0).tolist(),'pooled':binary_metrics(pp[:,0],pp[:,1]).tolist(),'batch_metrics':np.asarray(per_mode[mode]).tolist()}
        r.update(collisions=collisions,nonpast_history_records_at_batch_starts=ties)
        runs.append(r);predictions.append(pred)
    return runs,predictions

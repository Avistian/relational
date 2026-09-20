"""L095: full-release bipartite audit and deterministic recommendation experiment.

The walk is a course baseline, not a claimed published-model reproduction.
IDs are zero-based inside each separate node type. Ratings >=4 define likes.
"""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import HeteroData

ARCHIVE_URL='https://files.grouplens.org/datasets/movielens/ml-100k.zip'
ARCHIVE_SHA256='50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229'

def sha256(path):
    """Identify exact bytes; neither filename nor row count identifies a dataset."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def acquire(path):
    """Download directly from GroupLens; verify before reading. Do not redistribute data."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix('.partial')
        urllib.request.urlretrieve(ARCHIVE_URL,tmp)
        if sha256(tmp)!=ARCHIVE_SHA256:
            tmp.unlink();raise ValueError('Downloaded archive hash mismatch')
        tmp.replace(path)
    if sha256(path)!=ARCHIVE_SHA256:raise ValueError('Archive hash mismatch')
    return path

def read_release(path):
    """Read full data and all five official splits without extracting the archive."""
    with zipfile.ZipFile(acquire(path)) as z:
        arrays={};hashes={}
        for name in ['u.data']+[f'u{i}.{s}' for i in range(1,6) for s in ['base','test']]:
            raw=z.read('ml-100k/'+name);hashes[name]=hashlib.sha256(raw).hexdigest()
            a=np.loadtxt(io.BytesIO(raw),dtype=np.int64);a[:,:2]-=1;arrays[name]=a
    return arrays,hashes

def audit_release(arrays):
    """Recompute the README's complete counts and five-fold partition contract."""
    full=arrays['u.data'];identities=set(map(tuple,full));pairs=set(map(tuple,full[:,:2]))
    assert len(full)==len(identities)==len(pairs)==100000
    assert set(full[:,0])==set(range(943)) and set(full[:,1])==set(range(1682))
    assert np.all((full[:,2]>=1)&(full[:,2]<=5))
    counts=np.bincount(full[:,0],minlength=943);assert counts.min()>=20
    seen=set();folds=[]
    for fold in range(1,6):
        a,b=arrays[f'u{fold}.base'],arrays[f'u{fold}.test'];sa,sb=set(map(tuple,a)),set(map(tuple,b))
        assert len(a)==len(sa)==80000 and len(b)==len(sb)==20000
        assert not sa&sb and sa|sb==identities and not seen&sb
        seen|=sb;folds.append({'fold':fold,'base':len(a),'test':len(b),'disjoint':True,'reconstructs_full':True})
    assert seen==identities
    return {'status':'MATCH','target':'GroupLens ML-100K README: counts and u1-u5 partitions','ratings':len(full),'users':943,'items':1682,'minimum_user_ratings':int(counts.min()),'folds':folds,'five_test_sets_partition_full':True}

def build_graph(rows,n_users,n_items):
    """Build likes and its exact reverse from these fitting rows only; retain all nodes."""
    rows=np.asarray(rows)
    if rows.ndim!=2 or rows.shape[1]!=4:raise ValueError('Expected user,item,rating,timestamp rows')
    if not np.issubdtype(rows.dtype,np.integer):raise ValueError('Expected integer IDs and ratings')
    if len(rows) and (rows[:,0].min()<0 or rows[:,0].max()>=n_users or rows[:,1].min()<0 or rows[:,1].max()>=n_items):raise ValueError('ID outside typed node range')
    if len(set(map(tuple,rows[:,:2])))!=len(rows):raise ValueError('Repeated pair: choose an event policy first')
    if len(rows) and not np.all((rows[:,2]>=1)&(rows[:,2]<=5)):raise ValueError('Rating outside 1..5')
    data=HeteroData();data['user'].num_nodes=n_users;data['item'].num_nodes=n_items
    edges=torch.as_tensor(rows[rows[:,2]>=4,:2].T.copy(),dtype=torch.long)
    data['user','likes','item'].edge_index=edges
    data['item','rev_likes','user'].edge_index=edges.flip(0).clone()
    data.validate(raise_on_error=True)
    return data

def assert_boundary(graph,heldout):
    """Verify both directions and reject held-out targets in the message graph."""
    f=graph['user','likes','item'].edge_index.cpu().numpy().T
    r=graph['item','rev_likes','user'].edge_index.cpu().numpy().T[:,::-1]
    sf,sr=set(map(tuple,f)),set(map(tuple,r));h=set(map(tuple,np.asarray(heldout)[:,:2]))
    assert len(sf)==len(f) and len(sr)==len(r),'Duplicate graph edge'
    assert sf==sr,'Reverse relation differs from forward relation'
    assert not sf&h and not sr&h,'Held-out interaction leaked into graph'

def split_inner(base,seed=95,fraction=.1):
    """Course validation split within official base; official test is untouched."""
    if not 0<fraction<1 or len(base)<2:raise ValueError('Need nonempty fitting and validation rows')
    order=np.random.default_rng(seed).permutation(len(base));n=max(1,min(len(base)-1,int(len(base)*fraction)))
    return base[order[n:]].copy(),base[order[:n]].copy()

def walk_scores(graph):
    """Three typed steps user→item→user→item; zero-degree users fall back to popularity."""
    nu,ni=graph['user'].num_nodes,graph['item'].num_nodes
    b=np.zeros((nu,ni),dtype=np.float64)
    edge=graph['user','likes','item'].edge_index.cpu().numpy();b[edge[0],edge[1]]=1
    du=b.sum(1);di=b.sum(0)
    u_to_i=np.divide(b,du[:,None],out=np.zeros_like(b),where=du[:,None]>0)
    i_to_u=np.divide(b.T,di[:,None],out=np.zeros_like(b.T),where=di[:,None]>0)
    scores=(u_to_i@i_to_u)@u_to_i
    pop=di/di.sum() if di.sum() else np.full(ni,1/ni)
    scores[du==0]=pop
    return scores,pop

def rank_metrics(scores,observed,heldout,k=10):
    """Full-catalog user-macro Recall/NDCG; no sampled negatives; stable item-ID ties."""
    scores=np.asarray(scores);nu,ni=scores.shape
    if k<1 or not np.isfinite(scores).all():raise ValueError('Finite scores and positive k required')
    observed=np.asarray(observed);heldout=np.asarray(heldout)
    if set(map(tuple,observed[:,:2]))&set(map(tuple,heldout[:,:2])):raise ValueError('Seen target pair')
    blocked=[set() for _ in range(nu)];truth=[set() for _ in range(nu)]
    for u,i in observed[:,:2]:blocked[int(u)].add(int(i))
    for u,i,r,t in heldout:
        if r>=4:truth[int(u)].add(int(i))
    records=[];ids=np.arange(ni)
    for u in range(nu):
        if not truth[u]:continue
        candidates=ids[~np.isin(ids,list(blocked[u]))]
        top=candidates[np.lexsort((candidates,-scores[u,candidates]))[:k]]
        hit=np.array([i in truth[u] for i in top],dtype=float)
        dcg=float((hit/np.log2(np.arange(len(top))+2)).sum())
        ideal=float((1/np.log2(np.arange(min(k,len(truth[u])))+2)).sum())
        records.append({'user':u,'relevant':len(truth[u]),'candidates':len(candidates),'top_items':top.tolist(),'recall':float(hit.sum()/len(truth[u])),'ndcg':dcg/ideal})
    return {'recall':float(np.mean([x['recall'] for x in records])) if records else None,'ndcg':float(np.mean([x['ndcg'] for x in records])) if records else None,'users':len(records),'excluded_no_relevant':nu-len(records),'k':k},records

def run_experiment(arrays,output_dir=None):
    """Select mixture on inner validation; refit full official base; evaluate test once per arm."""
    runs=[]
    if output_dir is not None:Path(output_dir).mkdir(parents=True,exist_ok=True)
    for fold in range(1,6):
        base,test=arrays[f'u{fold}.base'],arrays[f'u{fold}.test'];fit,val=split_inner(base,seed=9500+fold)
        graph=build_graph(fit,943,1682);assert_boundary(graph,np.vstack([val,test]))
        w,p=walk_scores(graph);validation=[]
        for alpha in [0.,.5,1.]:
            metrics,_=rank_metrics(alpha*w+(1-alpha)*p[None,:],fit,val)
            validation.append({'alpha':alpha,**metrics})
        # Ties prefer the simpler popularity model (lower alpha).
        chosen=max(validation,key=lambda x:x['ndcg'])['alpha']
        graph=build_graph(base,943,1682);assert_boundary(graph,test);w,p=walk_scores(graph)
        models={};details={}
        for name,alpha in [('popularity',0.),('walk',1.),('validation_selected',chosen)]:
            models[name],details[name]=rank_metrics(alpha*w+(1-alpha)*p[None,:],base,test)
        entry={'fold':fold,'split_seed':9500+fold,'fit_rows':len(fit),'validation_rows':len(val),'refit_rows':len(base),'test_rows':len(test),'graph_like_edges':int(graph['user','likes','item'].edge_index.shape[1]),'alpha':chosen,'validation':validation,'test':models}
        if output_dir is not None:
            path=Path(output_dir)/f'fold-{fold}.json';path.write_text(json.dumps(details,indent=2)+'\n');entry['rankings_sha256']=sha256(path)
            split=Path(output_dir)/f'fold-{fold}-inner.npz';np.savez_compressed(split,fit=fit,validation=val);entry['inner_arrays_sha256']=hashlib.sha256(fit.tobytes()+val.tobytes()).hexdigest()
        runs.append(entry)
    summary={name:{metric:{'mean':float(np.mean([r['test'][name][metric] for r in runs])),'fold_sd':float(np.std([r['test'][name][metric] for r in runs],ddof=1))} for metric in ['recall','ndcg']} for name in ['popularity','walk','validation_selected']}
    return {'status':'MEASURED','scope':'Course ranking protocol on all five complete official ML-100K folds','full_paper_parity':'NOT_APPLICABLE_NO_MODEL_PAPER','historical_model_score_parity':'NOT_ESTABLISHED','runs':runs,'summary':summary,'uncertainty':'Fold SD is descriptive; overlapping training folds are not independent datasets.'}

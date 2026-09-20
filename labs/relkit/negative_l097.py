"""L097: typed, train-only negative sampling and full ML-100K ablation.

Course experiment using a BPR pairwise objective; not historical BPR reproduction.
"""
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ARCHIVE_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
ARCHIVE_SHA256 = '50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229'
CONFIG = {'folds': [1,2,3,4,5], 'seeds': [0,1,2], 'arms': ['uniform','degree','hard'],
          'epochs': 10, 'dimension': 32, 'batch_size': 2048, 'lr': .01,
          'l2': .0001, 'hard_pool': 4, 'sampled_distractors': 99, 'k': 10}

def load_release(path):
    """Fetch original bytes, then audit every row of all five official partitions."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix('.partial');urllib.request.urlretrieve(ARCHIVE_URL,tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=ARCHIVE_SHA256:
            tmp.unlink();raise ValueError('Downloaded archive hash mismatch')
        tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ARCHIVE_SHA256:raise ValueError('Archive hash mismatch')
    arrays={};hashes={}
    with zipfile.ZipFile(path) as z:
        for name in ['u.data']+[f'u{f}.{part}' for f in range(1,6) for part in ['base','test']]:
            raw=z.read('ml-100k/'+name);hashes[name]=hashlib.sha256(raw).hexdigest()
            a=np.loadtxt(io.BytesIO(raw),dtype=np.int64);a[:,:2]-=1;arrays[name]=a
    full=arrays['u.data'];truth=set(map(tuple,full));seen=set()
    assert len(full)==len(truth)==len(set(map(tuple,full[:,:2])))==100000
    assert set(full[:,0])==set(range(943)) and set(full[:,1])==set(range(1682))
    assert np.bincount(full[:,0]).min()==20
    for fold in range(1,6):
        a,b=arrays[f'u{fold}.base'],arrays[f'u{fold}.test'];sa,sb=set(map(tuple,a)),set(map(tuple,b))
        assert len(a)==len(sa)==80000 and len(b)==len(sb)==20000
        assert not sa&sb and sa|sb==truth and not seen&sb;seen|=sb
    assert seen==truth
    return arrays,hashes

def training_inputs(base,n_users=943,n_items=1682):
    """Only base rows enter masks, degrees and positives. Ratings >=4 are likes."""
    blocked=np.zeros((n_users,n_items),dtype=bool);blocked[base[:,0],base[:,1]]=True
    positive=base[base[:,2]>=4,:2].copy()
    degree=np.bincount(positive[:,1],minlength=n_items)
    return positive,blocked,degree

def proposal(blocked,degree,strategy):
    """TODO: return normalized q(item|user) over unobserved typed destinations.

    uniform uses unit weights; degree uses (like_degree+1)^0.75. Empty rows fail.
    This dense pedagogical implementation costs O(users*items), not production scale.
    """
    blocked=np.asarray(blocked);degree=np.asarray(degree,dtype=float)
    if blocked.ndim!=2 or blocked.dtype!=bool or degree.shape!=(blocked.shape[1],):raise ValueError('Invalid typed shapes')
    if not np.isfinite(degree).all() or (degree<0).any():raise ValueError('Invalid degrees')
    if strategy=='uniform':weights=np.ones(len(degree))
    elif strategy=='degree':weights=(degree+1)**.75
    else:raise ValueError('Unknown proposal')
    mass=(~blocked)*weights[None,:];totals=mass.sum(1,keepdims=True)
    if (totals==0).any():raise ValueError('No valid negative for at least one source')
    return mass/totals

def draw_negatives(users,q,rng):
    """TODO: one independent replacement draw per source, preserving user identity.

    Group inverse-CDF queries by user. Equal user/item indices are valid typed pairs.
    Duplicates across draws are allowed; positive collisions have probability zero.
    """
    users=np.asarray(users,dtype=np.int64)
    if users.ndim!=1 or ((users<0)|(users>=len(q))).any():raise ValueError('Invalid source IDs')
    out=np.empty(len(users),dtype=np.int64);uniforms=rng.random(len(users))
    for u in np.unique(users):
        locations=np.flatnonzero(users==u);cdf=np.cumsum(q[u]);cdf[-1]=1.
        out[locations]=np.searchsorted(cdf,uniforms[locations],side='right')
    return out

def pairwise_loss(positive_score,negative_score):
    """TODO: mean negative log sigmoid(score_positive - score_negative), stably."""
    return F.softplus(negative_score-positive_score).mean()

def fit(base,strategy,seed,config=CONFIG):
    """Visible fixed-budget MF trainer. Heldout data are not an argument.

    Shared initial weights and positive orders across arms; fresh negatives each epoch.
    Hard arm chooses the largest detached score from four uniform candidate draws.
    """
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    positive,blocked,degree=training_inputs(base)
    q=proposal(blocked,degree,'degree' if strategy=='degree' else 'uniform')
    user=torch.nn.Parameter(torch.randn(943,config['dimension'])*.1)
    item=torch.nn.Parameter(torch.randn(1682,config['dimension'])*.1)
    bias=torch.nn.Parameter(torch.zeros(1682))
    initial=hashlib.sha256(user.detach().numpy().tobytes()+item.detach().numpy().tobytes()).hexdigest()
    opt=torch.optim.Adam([user,item,bias],lr=config['lr'])
    order_rng=np.random.default_rng(9700+seed);negative_rng=np.random.default_rng(19700+seed)
    trace=[];negative_hash=hashlib.sha256();collisions=0;draws=0
    for epoch in range(config['epochs']):
        order=order_rng.permutation(len(positive));loss_sum=0.
        # Draw one (or four hard-pool) candidates per positive before batching.
        width=config['hard_pool'] if strategy=='hard' else 1
        repeated=np.repeat(positive[:,0],width)
        candidates=draw_negatives(repeated,q,negative_rng).reshape(-1,width)
        assert not blocked[repeated,candidates.ravel()].any()
        for start in range(0,len(order),config['batch_size']):
            ids=order[start:start+config['batch_size']];u,i=positive[ids].T
            cand=torch.from_numpy(candidates[ids]);ut=torch.from_numpy(u);it=torch.from_numpy(i)
            if strategy=='hard':
                with torch.no_grad():
                    scores=(user[ut,None,:]*item[cand]).sum(-1)+bias[cand]
                    j=cand[torch.arange(len(ids)),scores.argmax(1)]
            else:j=cand[:,0]
            negative_hash.update(j.numpy().tobytes());draws+=len(ids)
            collisions+=int(blocked[u,j.numpy()].sum())
            sp=(user[ut]*item[it]).sum(1)+bias[it]
            sn=(user[ut]*item[j]).sum(1)+bias[j]
            reg=(user[ut].square().sum(1)+item[it].square().sum(1)+item[j].square().sum(1)+bias[it].square()+bias[j].square()).mean()
            loss=pairwise_loss(sp,sn)+config['l2']*reg
            opt.zero_grad();loss.backward();opt.step();loss_sum+=loss.item()*len(ids)
        trace.append(loss_sum/len(positive))
    with torch.no_grad():scores=(user@item.T+bias).numpy()
    assert np.isfinite(scores).all() and collisions==0
    return scores,blocked,{'loss_by_epoch':trace,'positive_rows':len(positive),'negative_draws':draws,'known_positive_collisions':collisions,'initial_sha256':initial,'negatives_sha256':negative_hash.hexdigest()}

def evaluate(scores,blocked,test,seed,k=10,distractors=99):
    """Same full-catalog rule for every arm; separate fixed sampled-candidate diagnostic.

    Binary relevant = test rating >=4. Exclude base-rated items. Macro over users
    with >=1 relevant item. Ties use ascending item ID. Full and sampled share scores.
    """
    truth={}
    for u,i,rating,_ in test:
        if rating>=4:truth.setdefault(int(u),set()).add(int(i))
    records=[];totals=[]
    for u,relevant in sorted(truth.items()):
        candidates=np.flatnonzero(~blocked[u]);rel=sorted(relevant)
        assert set(rel)<=set(candidates),'Heldout/base overlap'
        negatives=np.array([i for i in candidates if i not in relevant],dtype=np.int64)
        rng=np.random.default_rng(seed*10000+u)
        sampled=np.sort(np.concatenate([np.array(rel),rng.choice(negatives,min(distractors,len(negatives)),replace=False)]))
        row={'user':u,'relevant':rel,'full_candidates':len(candidates),'sampled_candidates':len(sampled)};values=[]
        for label,pool in [('full',candidates),('sampled',sampled)]:
            top=pool[np.lexsort((pool,-scores[u,pool]))[:k]]
            hit=np.isin(top,rel);recall=float(hit.sum()/len(rel))
            ndcg=float(np.sum(hit/np.log2(np.arange(len(top))+2))/np.sum(1/np.log2(np.arange(min(k,len(rel)))+2)))
            row[label+'_top']=top.tolist();row[label+'_recall']=recall;row[label+'_ndcg']=ndcg
            values.extend([recall,ndcg])
        records.append(row);totals.append(values)
    if not records:raise ValueError('No evaluable users')
    keys=['full_recall','full_ndcg','sampled_recall','sampled_ndcg']
    result=dict(zip(keys,map(float,np.mean(totals,axis=0))))
    result.update({'evaluated_users':len(records),'excluded_users':len(scores)-len(records)})
    return result,records

def run_suite(archive,output_dir):
    """All five official folds x three seeds x three samplers, no subsampling/resume."""
    start=time.monotonic();arrays,hashes=load_release(archive);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for fold in CONFIG['folds']:
        base,test=arrays[f'u{fold}.base'],arrays[f'u{fold}.test']
        for seed in CONFIG['seeds']:
            for arm in CONFIG['arms']:
                scores,blocked,training=fit(base,arm,seed)
                metrics,records=evaluate(scores,blocked,test,970+fold,k=CONFIG['k'],distractors=CONFIG['sampled_distractors'])
                name=f'fold{fold}-seed{seed}-{arm}.json';payload=json.dumps(records,separators=(',',':'))+'\n'
                (out/name).write_text(payload)
                row={'fold':fold,'seed':seed,'arm':arm,**metrics,**training,'records_file':name,'records_sha256':hashlib.sha256(payload.encode()).hexdigest()};rows.append(row)
                print(f"fold={fold} seed={seed} arm={arm}: full NDCG={metrics['full_ndcg']:.5f}, sampled={metrics['sampled_ndcg']:.5f}",flush=True)
    summary={}
    for arm in CONFIG['arms']:
        summary[arm]={}
        for metric in ['full_recall','full_ndcg','sampled_recall','sampled_ndcg']:
            folds=[float(np.mean([r[metric] for r in rows if r['arm']==arm and r['fold']==f])) for f in CONFIG['folds']]
            summary[arm][metric]={'mean':float(np.mean(folds)),'fold_means':folds,'fold_sd':float(np.std(folds,ddof=1))}
    return {'status':'MEASURED','scope':'complete course experiment; not historical BPR reproduction','config':CONFIG,'archive_sha256':ARCHIVE_SHA256,'member_sha256':hashes,'runs':rows,'summary':summary,'seconds':time.monotonic()-start,'paper_parity':'NOT_ESTABLISHED','learner':'PENDING_WRITTEN_DEFENSE'}

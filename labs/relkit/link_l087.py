"""L087: visible link-prediction mechanisms and full released-data heuristic reconstruction."""
from pathlib import Path
import copy
import hashlib
import json
import time
import urllib.request
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata, friedmanchisquare, studentized_range
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F


def canonical_edges(edges, n):
    """One lexicographically ordered u<v row per undirected edge; remove loops."""
    e=np.asarray(edges,dtype=np.int64).reshape(-1,2)
    if np.any(e<0) or np.any(e>=n):raise ValueError('Node ID outside graph')
    e=np.sort(e,axis=1)
    return np.unique(e[e[:,0]!=e[:,1]],axis=0)


def split_edges(edges,n,seed=87,val_fraction=.1,test_fraction=.1):
    """TODO: split canonical pairs before adding reverse message-passing edges."""
    if min(val_fraction,test_fraction)<0 or val_fraction+test_fraction>=1:raise ValueError('Invalid fractions')
    e=canonical_edges(edges,n);e=e[np.random.default_rng(seed).permutation(len(e))]
    nv=int(len(e)*val_fraction);nt=int(len(e)*test_fraction)
    return e[nv+nt:],e[:nv],e[nv:nv+nt]


def sample_non_edges(positives,n,count,rng):
    """Uniform unordered negatives without replacement; all known positives excluded.

    Enumerating the complement is suitable for these small graphs, not web-scale graphs.
    """
    e=canonical_edges(positives,n);u,v=np.triu_indices(n,1)
    keep=~np.isin(u*n+v,e[:,0]*n+e[:,1]);candidates=np.column_stack((u[keep],v[keep]))
    if count>len(candidates):raise ValueError('Not enough distinct non-edges')
    return candidates[rng.choice(len(candidates),count,replace=False)]


def adjacency(edges,n):
    """Binary symmetric CSR; num_nodes preserves isolated vertices."""
    e=canonical_edges(edges,n);u,v=e.T
    return sp.csr_matrix((np.ones(2*len(e)),(np.r_[u,v],np.r_[v,u])),shape=(n,n))


def edge_logits(z,pairs):
    """TODO: return one dot-product logit per [u,v] row, preserving gradients."""
    return (z[pairs[:,0]]*z[pairs[:,1]]).sum(dim=-1)


def ranking_metrics(positive,negative,k=10):
    """TODO: average optimistic/pessimistic rank per query, then MRR and Hits@k.

    positive: [Q]; negative: [Q,K]; a positive is ranked against its own K candidates.
    """
    positive=np.asarray(positive);negative=np.asarray(negative)
    if positive.ndim!=1 or negative.ndim!=2 or len(positive)!=len(negative):raise ValueError('Expected [Q], [Q,K]')
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():raise ValueError('Nonfinite score')
    rank=1+(negative>positive[:,None]).sum(1)+.5*(negative==positive[:,None]).sum(1)
    return {'ranks':rank.tolist(),'mrr':float(np.mean(1/rank)),'hits_at_k':float(np.mean(rank<=k)),'k':k,'negatives_per_query':negative.shape[1]}


def heuristic_scores(a,pairs):
    """CN=A A; AA=A diag(1/log degree) A; RA=A diag(1/degree) A."""
    degree=np.asarray(a.sum(1)).ravel();cn=np.ones(len(degree));aa=np.zeros(len(degree));ra=np.zeros(len(degree))
    aa[degree>1]=1/np.log(degree[degree>1]);ra[degree>0]=1/degree[degree>0]
    return {name:np.asarray((a@sp.diags(w)@a)[pairs[:,0],pairs[:,1]]).ravel() for name,w in [('CN',cn),('AA',aa),('RA',ra)]}


def drnl_subgraph(a,u,v,hops=1):
    """Enclose both roots, remove target, then label distances with opposite root deleted.

    DRNL labels endpoints 1, unreachable nodes 0. This is the SEAL labeling primitive,
    not a SEAL classifier or reproduction of its trained DGCNN.
    """
    nodes={u,v};frontier={u,v}
    for _ in range(hops):
        neighbors=set().union(*(set(a[i].indices) for i in frontier));frontier=neighbors-nodes;nodes|=frontier
    nodes=[u,v]+sorted(nodes-{u,v});s=a[nodes][:,nodes].tolil();s[0,1]=s[1,0]=0;s=s.tocsr()
    keep_u=[0]+list(range(2,len(nodes)));keep_v=[1]+list(range(2,len(nodes)))
    du=shortest_path(s[keep_u][:,keep_u],directed=False,unweighted=True,indices=0)[1:]
    dv=shortest_path(s[keep_v][:,keep_v],directed=False,unweighted=True,indices=0)[1:]
    labels=np.zeros(len(nodes),dtype=int);labels[:2]=1
    finite=np.isfinite(du)&np.isfinite(dv);d=du[finite]+dv[finite];q=np.floor(d/2);r=np.mod(d,2)
    labels[np.flatnonzero(finite)+2]=(1+np.minimum(du[finite],dv[finite])+q*(q+r-1)).astype(int)
    return s,labels,nodes


def load_graph(name,manifest,root):
    """Download the commit-pinned bytes only if absent; check SHA256 before loading."""
    entry=next(r for r in manifest['files'] if r['path']==f'MATLAB/data/{name}.mat')
    p=Path(root)/entry['path'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():p.write_bytes(urllib.request.urlopen(entry['url'],timeout=60).read())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Data hash mismatch: '+name)
    a=sp.csr_matrix(loadmat(p)['net']);a=a.maximum(a.T);a.setdiag(0);a.eliminate_zeros();a.data[:]=1
    return canonical_edges(np.column_stack(sp.triu(a,1).nonzero()),a.shape[0]),a.shape[0]


def source_split(edges,n,seed):
    """Reconstruct connected=false DivideNet with MT19937 uniforms and column-major order.

    NumPy permutation in negative sampling differs from MATLAB randperm. Preserve actual
    splits/candidates in artifacts; equal seed numbers do not imply historical identity.
    """
    rng=np.random.RandomState(seed);e=edges[np.lexsort((edges[:,0],edges[:,1]))].copy();test=[]
    for _ in range(int(np.floor((1-.9)*len(e)))):
        i=int(np.ceil(rng.random_sample()*len(e)))-1;test.append(e[i]);e=np.delete(e,i,axis=0)
    e=e[np.lexsort((e[:,0],e[:,1]))];test=np.array(test);test=test[np.lexsort((test[:,0],test[:,1]))]
    # Match the uniform complement policy, using a modern permutation algorithm.
    neg=sample_non_edges(edges,n,len(edges),rng)
    return e,neg[:len(e)],test,neg[len(e):]


def run_paper(manifest,root,out,datasets=None,seeds=range(1,11)):
    """Full Table1 CN/AA/RA columns: all eight graphs, 10 independent random splits."""
    datasets=datasets or ['USAir','NS','PB','Yeast','Celegans','Power','Router','Ecoli']
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];start=time.time()
    for name in datasets:
        e,n=load_graph(name,manifest,root)
        for seed in seeds:
            train,train_neg,test,neg=source_split(e,n,seed);pairs=np.vstack((test,neg));labels=np.r_[np.ones(len(test)),np.zeros(len(neg))]
            scores=heuristic_scores(adjacency(train,n),pairs)
            p=out/f'{name}-{seed}.npz';np.savez_compressed(p,train=train,train_neg=train_neg,test=test,test_neg=neg,**scores)
            rows.append({'dataset':name,'seed':seed,'nodes':n,'edges':len(e),'train_edges':len(train),'test_edges':len(test),'auc':{k:float(roc_auc_score(labels,v)) for k,v in scores.items()},'artifact':p.name,'artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        print(name,'complete',flush=True)
    summary={name:{m:{'mean':float(np.mean([r['auc'][m] for r in rows if r['dataset']==name])),'sample_sd':float(np.std([r['auc'][m] for r in rows if r['dataset']==name],ddof=1))} for m in ['CN','AA','RA']} for name in datasets}
    stats={}
    if len(datasets)>=3:
        means=np.array([[summary[n][m]['mean'] for m in ['CN','AA','RA']] for n in datasets]);ranks=np.array([rankdata(-x) for x in means]);f=friedmanchisquare(*means.T)
        stats={'methods':['CN','AA','RA'],'mean_ranks':ranks.mean(0).tolist(),'friedman_statistic':float(f.statistic),'friedman_p':float(f.pvalue),'nemenyi_cd_05':float(studentized_range.ppf(.95,3,np.inf)/np.sqrt(2)*np.sqrt(3*4/(6*len(datasets)))),'unit':'one equally weighted dataset; means over seeds'}
    return {'target':'Zhang & Chen 2018 Table1 CN/AA/RA, eight graphs, ten splits','status':'FULL_PROTOCOL_RECONSTRUCTION' if len(rows)==80 else 'PARTIAL_RECONSTRUCTION','historical_parity':'INCOMPARABLE','deviations':['NumPy complement permutation replaces MATLAB randperm; exact historical candidates unverified','Pinned current source revision not established as original paper-run revision'],'full_SEAL_training':'NOT_RUN','runs':rows,'summary':summary,'statistics':stats,'seconds':time.time()-start,'source_revision':manifest['revision']}


def normalized_support(a):
    """GCN symmetric normalization on context edges plus self-loops only."""
    a=a+sp.eye(a.shape[0]);d=np.asarray(a.sum(1)).ravel()**-.5;s=(sp.diags(d)@a@sp.diags(d)).tocoo()
    return torch.sparse_coo_tensor(np.vstack((s.row,s.col)),s.data.astype('float32'),s.shape).coalesce()


class LinkGCN(nn.Module):
    """Teaching encoder: trainable node IDs → GCN32/ReLU → GCN16; shared across queries."""
    def __init__(self,n):
        super().__init__();self.ids=nn.Embedding(n,32);self.w1=nn.Linear(32,32,bias=False);self.w2=nn.Linear(32,16,bias=False)
        nn.init.normal_(self.ids.weight,std=.1)
    def forward(self,s):
        h=F.relu(torch.sparse.mm(s,self.w1(self.ids.weight)))
        return torch.sparse.mm(s,self.w2(h))


def ranking_candidates(positive,all_edges,n,seed,k=50):
    """Both query orientations; K distinct destinations per source, all positives filtered.

    This retrospective closed-world filtering is an evaluation rule, not future information
    supplied to the encoder. A real recommendation task needs timestamp-valid eligibility.
    """
    rng=np.random.default_rng(seed);a=adjacency(all_edges,n);pos=np.vstack((positive,positive[:,::-1]));neg=[]
    for u,v in pos:
        available=np.setdiff1d(np.arange(n),np.r_[u,a[u].indices])
        if len(available)<k:raise ValueError('Candidate pool smaller than K; specify a smaller K')
        dest=rng.choice(available,k,replace=False);neg.append(np.column_stack((np.full(k,u),dest)))
    return pos,np.array(neg)


def train_link(edges,n,seed=87,epochs=150):
    """Separate teaching protocol: 80/10/10; half train context, half supervision.

    Fixed full-batch Adam(.01), no HPO/dropout; validation AUC selects checkpoint.
    Graph and negatives are frozen. Test read only after checkpoint selection.
    """
    torch.set_num_threads(1);torch.manual_seed(seed)
    train,val,test=split_edges(edges,n,seed);half=len(train)//2;context,positive=train[:half],train[half:]
    rng=np.random.default_rng(seed+1000);negs=sample_non_edges(edges,n,len(positive)+len(val)+len(test),rng)
    train_neg=negs[:len(positive)];val_neg=negs[len(positive):len(positive)+len(val)];test_neg=negs[len(positive)+len(val):]
    s=normalized_support(adjacency(context,n));model=LinkGCN(n);opt=torch.optim.Adam(model.parameters(),lr=.01)
    pairs=torch.tensor(np.vstack((positive,train_neg)));y=torch.tensor(np.r_[np.ones(len(positive)),np.zeros(len(train_neg))],dtype=torch.float32)
    vp=torch.tensor(np.vstack((val,val_neg)));vy=np.r_[np.ones(len(val)),np.zeros(len(val_neg))]
    best=-np.inf;trace=[]
    for epoch in range(epochs):
        model.train();opt.zero_grad();loss=F.binary_cross_entropy_with_logits(edge_logits(model(s),pairs),y);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():score=roc_auc_score(vy,edge_logits(model(s),vp).numpy())
        trace.append({'epoch':epoch+1,'loss':float(loss.detach()),'val_auc':float(score)})
        if score>best:best=score;state=copy.deepcopy(model.state_dict());selected=epoch+1
    model.load_state_dict(state);model.eval();rp,rn=ranking_candidates(test,edges,n,seed+2000)
    with torch.no_grad():
        z=model(s);tp=torch.tensor(np.vstack((test,test_neg)));ts=edge_logits(z,tp).numpy();ps=edge_logits(z,torch.tensor(rp)).numpy();ns=edge_logits(z,torch.tensor(rn.reshape(-1,2))).numpy().reshape(len(rp),-1)
    r=ranking_metrics(ps,ns)
    result={'status':'TEACHING_RUN','seed':seed,'epochs':epochs,'selected_epoch':selected,'validation_auc':best,'test_auc':float(roc_auc_score(np.r_[np.ones(len(test)),np.zeros(len(test_neg))],ts)),'ranking':r,'trace':trace,'protocol':'80/10/10; half train used only as context; both query directions; 50 filtered negatives/query','paper_parity':'INCOMPARABLE'}
    arrays={'context':context,'train_pos':positive,'train_neg':train_neg,'val_pos':val,'val_neg':val_neg,'test_pos':test,'test_neg':test_neg,'ranking_pos':rp,'ranking_neg':rn,'positive_scores':ps,'negative_scores':ns,'auc_scores':ts,'embeddings':z.numpy()}
    return result,arrays

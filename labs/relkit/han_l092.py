"""Visible PyTorch reconstruction of Jhy1993/HAN; see l092-reproduction.md.
Written for teaching from the pinned release equations and training recipe.
Paper-global and released per-node semantic attention are explicit alternatives.
"""
import copy
import hashlib
import json
import pickle
import time
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


def metapath_reachability(relations):
    """Multiply typed incidence matrices, then retain distinct endpoints, not walk counts."""
    reach=sp.csr_matrix(relations[0],dtype=np.float32)
    for relation in relations[1:]:
        reach=reach @ sp.csr_matrix(relation,dtype=np.float32)
        reach.eliminate_zeros()
        reach.data[:]=1
    reach.data[:]=1
    return reach


def neighbor_attention(h,edge,left,right,dropout,training):
    """Receiver-row softmax of LeakyReLU(left_i+right_j), followed by weighted sum."""
    receiver,sender=edge
    scores=F.leaky_relu(left[receiver]+right[sender],negative_slope=.2)
    row_max=torch.full((len(h),),-torch.inf,device=h.device,dtype=h.dtype)
    row_max.scatter_reduce_(0,receiver,scores.detach(),reduce='amax',include_self=True)
    exp=(scores-row_max[receiver]).exp()
    denom=torch.zeros_like(row_max).index_add(0,receiver,exp)
    alpha=exp/denom[receiver]
    alpha=F.dropout(alpha,dropout,training)
    support=torch.sparse_coo_tensor(edge,alpha,(len(h),len(h)),is_coalesced=True)
    return torch.sparse.mm(support,h)


def semantic_fusion(z,w,b,q,mode):
    """z: [nodes,paths,64]; w:[64,128]; q:[128]. Softmax always over paths."""
    scores=torch.tanh(z@w+b)@q
    if mode=='paper_global':
        scores=scores.mean(0,keepdim=True).expand(len(z),-1)
    elif mode!='release_node':
        raise ValueError('mode must be paper_global or release_node')
    beta=scores.softmax(dim=1)
    return (z*beta.unsqueeze(-1)).sum(dim=1),beta


def masked_loss(logits,labels,indices):
    return F.cross_entropy(logits[indices],labels[indices])


class AttentionHead(nn.Module):
    """Release head: three dropout sites, affine attention scores, post-sum bias, ELU."""
    def __init__(self,features,width=8,dropout=.6):
        super().__init__();self.dropout=dropout
        self.projection=nn.Linear(features,width,bias=False)
        self.left=nn.Linear(width,1);self.right=nn.Linear(width,1)
        self.bias=nn.Parameter(torch.zeros(width))
        for layer in [self.projection,self.left,self.right]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:nn.init.zeros_(layer.bias)
    def forward(self,x,edge):
        h=self.projection(F.dropout(x,self.dropout,self.training))
        left=self.left(h).squeeze(-1);right=self.right(h).squeeze(-1)
        values=F.dropout(h,self.dropout,self.training)
        return F.elu(neighbor_attention(values,edge,left,right,self.dropout,self.training)+self.bias)


class HAN(nn.Module):
    """Two path-specific banks of eight heads; 64→128 semantic scorer; 64→3 readout."""
    def __init__(self,features,classes=3,mode='release_node',paths=2):
        super().__init__();self.mode=mode
        self.branches=nn.ModuleList([nn.ModuleList([AttentionHead(features) for _ in range(8)]) for _ in range(paths)])
        self.w=nn.Parameter(torch.randn(64,128)*.1)
        self.b=nn.Parameter(torch.randn(128)*.1)
        self.q=nn.Parameter(torch.randn(128)*.1)
        self.readout=nn.Linear(64,classes)
        nn.init.xavier_uniform_(self.readout.weight);nn.init.zeros_(self.readout.bias)
    def forward(self,x,edges):
        if len(edges)!=len(self.branches):raise ValueError('One support required per branch')
        z=torch.stack([torch.cat([head(x,edge) for head in branch],dim=1) for branch,edge in zip(self.branches,edges)],dim=1)
        embedding,beta=semantic_fusion(z,self.w,self.b,self.q,self.mode)
        return self.readout(embedding),embedding,beta


def load_acm(root,manifest):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'ACM3025.pkl'
    spec=manifest['data']
    if not path.exists():
        with urllib.request.urlopen(spec['url'],timeout=120) as response:path.write_bytes(response.read())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=spec['sha256']:raise ValueError('Dataset checksum mismatch; refusing deserialization')
    # Only deserialize bytes from the fixed, verified DGL dataset digest.
    with path.open('rb') as f:data=pickle.load(f)
    x=torch.tensor(data['feature'].toarray(),dtype=torch.float32)
    y=torch.tensor(np.asarray(data['label'].toarray()).argmax(1),dtype=torch.long)
    edges=[];counts=[]
    for key in ['PAP','PLP']:
        support=metapath_reachability([data[key]])
        support.setdiag(1);support.eliminate_zeros();support.sort_indices()
        a=support.tocoo();edge=torch.tensor(np.stack([a.row,a.col]),dtype=torch.long)
        edges.append(edge);counts.append(support.nnz)
    indices=[torch.tensor(data[key].ravel(),dtype=torch.long) for key in ['train_idx','val_idx','test_idx']]
    train,val,test=indices
    assert x.shape==(3025,1870) and len(torch.unique(torch.cat(indices)))==3025
    assert [len(v) for v in indices]==[600,300,2125]
    meta={'sha256':digest,'nodes':len(x),'features':x.shape[1],'edges_PAP_PSP':counts,'split':[len(v) for v in indices],
          'split_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in indices)).hexdigest(),
          'provenance':'DGL ACM3025 mirror; original authors MAT byte identity not established'}
    return x,y,edges,train,val,test,meta


def knn_probe(embedding,labels,node_ids,seed=1000):
    """Release jhyexp.py: cumulative shuffle; first floor(n*f) train, remaining test, k=5."""
    rng=np.random.RandomState(seed);x=np.asarray(embedding);y=np.asarray(labels);ids=np.asarray(node_ids)
    records=[]
    for fraction in [.2,.4,.6,.8]:
        for repeat in range(10):
            permutation=rng.permutation(len(x));x=x[permutation];y=y[permutation];ids=ids[permutation]
            cut=int(len(x)*fraction)
            model=KNeighborsClassifier(n_neighbors=5);model.fit(x[:cut],y[:cut]);prediction=model.predict(x[cut:])
            records.append({'fraction':fraction,'repeat':repeat,'macro_f1':float(f1_score(y[cut:],prediction,average='macro')),
                'micro_f1':float(f1_score(y[cut:],prediction,average='micro')),'train_ids':ids[:cut].tolist(),
                'test_ids':ids[cut:].tolist(),'predictions':prediction.tolist()})
    return records


def train_acm(data,seed=0,epochs=200,mode='release_node',progress=False):
    torch.manual_seed(seed);np.random.seed(seed)
    x,y,edges,train,val,test,meta=data
    model=HAN(x.shape[1],mode=mode)
    # Upstream exact-name exclusion does not match scoped TF variable names: all parameters receive L2.
    optimizer=torch.optim.Adam(model.parameters(),lr=.005,eps=1e-8)
    best_acc=0.;best_loss=float('inf');waiting=0;checkpoint=None;selected_epoch=None;trace=[]
    started=time.perf_counter()
    for epoch in range(epochs):
        model.train();optimizer.zero_grad();logits,_,_=model(x,edges)
        loss=masked_loss(logits,y,train)
        regularizer=.001*.5*sum(p.square().sum() for p in model.parameters())
        (loss+regularizer).backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            logits,_,_=model(x,edges);vl=float(masked_loss(logits,y,val));va=float((logits[val].argmax(1)==y[val]).float().mean())
        trace.append({'epoch':epoch+1,'train_ce':float(loss.detach()),'validation_ce':vl,'validation_accuracy':va})
        # Preserve upstream OR-reset / AND-save; equality counts as improvement.
        if va>=best_acc or vl<=best_loss:
            if va>=best_acc and vl<=best_loss:
                checkpoint=copy.deepcopy(model.state_dict());selected_epoch=epoch+1
            best_acc=max(best_acc,va);best_loss=min(best_loss,vl);waiting=0
        else:
            waiting+=1
            if waiting==100:break
        if progress and (epoch==0 or (epoch+1)%20==0):print(seed,mode,epoch+1,round(vl,4),round(va,4),round(time.perf_counter()-started,1),flush=True)
    assert checkpoint is not None
    model.load_state_dict(checkpoint);model.eval()
    with torch.no_grad():
        logits,z,beta=model(x,edges);prediction=logits[test].argmax(1)
    result={'seed':seed,'mode':mode,'epochs_completed':len(trace),'epoch_ceiling':epochs,'selected_epoch':selected_epoch,
       'early_stop_patience':100,'trace':trace,'test_accuracy':float((prediction==y[test]).float().mean()),
       'test_predictions':prediction.tolist(),'test_ids':test.tolist(),'test_labels':y[test].tolist(),
       'semantic_mean_all_nodes':beta.mean(0).tolist(),'semantic_mean_test':beta[test].mean(0).tolist(),
       'semantic_per_test_node':beta[test].tolist(),'seconds':time.perf_counter()-started,
       'knn':knn_probe(z[test].numpy(),y[test].numpy(),test.numpy(),1000+seed)}
    return result,model


def summarize(runs,meta):
    targets={.2:(.8940,.8922),.4:(.8979,.8964),.6:(.8951,.8933),.8:(.9063,.9054)}
    rows=[]
    for fraction,(macro,micro) in targets.items():
        records=[r for run in runs for r in run['knn'] if r['fraction']==fraction]
        rows.append({'fraction':fraction,'macro_mean':float(np.mean([r['macro_f1'] for r in records])),
         'micro_mean':float(np.mean([r['micro_f1'] for r in records])),
         'macro_split_sd':float(np.std([r['macro_f1'] for r in records],ddof=1)),
         'micro_split_sd':float(np.std([r['micro_f1'] for r in records],ddof=1)),
         'paper_macro':macro,'paper_micro':micro,'descriptive_tolerance_pp':2,
         'verdict':'INCOMPARABLE','reason':'release semantics, mirrored data provenance, modern backend and unavailable historical seeds'})
    return {'status':'FULL_RELEASE_SCHEDULE_EXECUTED' if all(r['epoch_ceiling']==200 for r in runs) else 'DIAGNOSTIC',
       'data':meta,'training_runs':len(runs),'knn_per_training_run':40,'table3_acm':rows,'runs':runs,
       'full_paper_parity':'NOT_ESTABLISHED','historical_parity':'INCOMPARABLE','remaining_datasets':'DBLP and IMDB NOT_RUN'}

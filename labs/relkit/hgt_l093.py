"""L093: visible HGT release port and OAG Paper-Field L2 training.
Reference: acbull/pyHGT revision85eaccd, MIT (archived in labs/sources/hgt-l093).
Paper Table2 parity is NOT_ESTABLISHED; see l093-reproduction.md.
"""
from pathlib import Path
from collections import defaultdict
import copy,hashlib,json,math,platform,time,urllib.request
import numpy as np
import pandas as pd
import dill
import torch
from torch import nn
from torch.nn import functional as F


def receiver_softmax(logits, target, nodes):
    """E×heads -> E×heads, normalize ALL incoming edges per receiver/head."""
    ix=target[:,None].expand_as(logits)
    maxima=logits.new_full((nodes,logits.shape[1]),-torch.inf)
    maxima.scatter_reduce_(0,ix,logits.detach(),reduce='amax',include_self=True)
    weights=(logits-maxima[target]).exp()
    denom=logits.new_zeros((nodes,logits.shape[1])).scatter_add_(0,ix,weights)
    return weights/denom[target].clamp_min(torch.finfo(logits.dtype).tiny)


def relation_heads(q,k,v,attention_matrix,message_matrix,prior):
    """Edge-selected q/k/v:E×H×D; relation matrices:E×H×D×D; prior:E×H."""
    key=torch.einsum('ehd,ehdf->ehf',k,attention_matrix)
    score=(q*key).sum(-1)*prior/math.sqrt(q.shape[-1])
    message=torch.einsum('ehd,ehdf->ehf',v,message_matrix)
    return score,message


def temporal_basis(index,width,mode='release'):
    """Release: paired frequency and 1/sqrt(width); paper: Eq6/7 as printed."""
    if width%2:raise ValueError('Temporal width must be even')
    if mode not in ('release','paper'):raise ValueError(mode)
    j=torch.arange(0,width,2,device=index.device,dtype=torch.float32)
    index=index.to(torch.float32).reshape(-1,1)
    out=index.new_empty((len(index),width))
    out[:,0::2]=torch.sin(index*torch.exp(-math.log(10000)*j/width))
    cosine_j=j if mode=='release' else j+1
    out[:,1::2]=torch.cos(index*torch.exp(-math.log(10000)*cosine_j/width))
    return out/math.sqrt(width) if mode=='release' else out


class RelativeTime(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.emb=nn.Embedding(240,width)
        with torch.no_grad():self.emb.weight.copy_(temporal_basis(torch.arange(240),width))
        # The release sets emb.requires_grad=False, not emb.weight.requires_grad=False.
        # Preserve its actual trainable table. This is a documented paper/release gap.
        self.lin=nn.Linear(width,width)
    def forward(self,x,index):
        if index.numel() and (index.min()<0 or index.max()>=240):
            raise ValueError('Release RTE index outside [0,239]; do not silently clip time gaps')
        return x+self.lin(self.emb(index))


class HGTLayer(nn.Module):
    def __init__(self,width,types,relations,heads,dropout=.2,use_norm=True,use_rte=True):
        super().__init__()
        if width%heads:raise ValueError('width must divide heads')
        self.width=width;self.heads=heads;self.d=width//heads;self.use_norm=use_norm;self.use_rte=use_rte
        self.k_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(types)])
        self.q_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(types)])
        self.v_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(types)])
        self.a_linears=nn.ModuleList([nn.Linear(width,width) for _ in range(types)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(types)] if use_norm else [])
        self.relation_pri=nn.Parameter(torch.ones(relations,heads))
        self.relation_att=nn.Parameter(torch.empty(relations,heads,self.d,self.d))
        self.relation_msg=nn.Parameter(torch.empty(relations,heads,self.d,self.d))
        self.skip=nn.Parameter(torch.ones(types));self.drop=nn.Dropout(dropout)
        if use_rte:self.emb=RelativeTime(width)
        # PyG glorot uses the last two dimensions, including on a 4D parameter.
        for w in (self.relation_att,self.relation_msg):nn.init.uniform_(w,-math.sqrt(3/self.d),math.sqrt(3/self.d))
        self.att=None
    def forward(self,x,node_type,edge_index,edge_type,edge_time):
        src,dst=edge_index;source=x[src];target=x[dst]
        if self.use_rte:source=self.emb(source,edge_time)
        q=torch.zeros_like(target);k=torch.zeros_like(source);v=torch.zeros_like(source)
        for t in range(len(self.k_linears)):
            sm=node_type[src]==t;tm=node_type[dst]==t
            k[sm]=self.k_linears[t](source[sm]);v[sm]=self.v_linears[t](source[sm]);q[tm]=self.q_linears[t](target[tm])
        q=q.reshape(-1,self.heads,self.d);k=k.reshape_as(q);v=v.reshape_as(q)
        # Relation grouping avoids an E×H×D×D allocation at paper width.
        scores=q.new_zeros((len(src),self.heads));messages=q.new_zeros(q.shape)
        for r in range(len(self.relation_pri)):
            mask=edge_type==r
            if mask.any():
                count=int(mask.sum())
                scores[mask],messages[mask]=relation_heads(q[mask],k[mask],v[mask],self.relation_att[r].expand(count,-1,-1,-1),self.relation_msg[r].expand(count,-1,-1,-1),self.relation_pri[r].expand(count,-1))
        self.att=receiver_softmax(scores,dst,len(x))
        aggregate=x.new_zeros((len(x),self.width))
        aggregate.index_add_(0,dst,(messages*self.att[:,:,None]).reshape(-1,self.width))
        aggregate=F.gelu(aggregate);out=torch.zeros_like(x)
        for t in range(len(self.a_linears)):
            mask=node_type==t;a=torch.sigmoid(self.skip[t])
            z=a*self.drop(self.a_linears[t](aggregate[mask]))+(1-a)*x[mask]
            out[mask]=self.norms[t](z) if self.use_norm else z
        return out


class RGCNLayer(nn.Module):
    """Local comparison: per-relation mean, then sum relations and learned self path."""
    def __init__(self,width,relations):
        super().__init__();self.weights=nn.Parameter(torch.empty(relations-1,width,width));self.self_linear=nn.Linear(width,width)
        for w in self.weights:nn.init.xavier_uniform_(w)
    def forward(self,x,node_type,edge_index,edge_type,edge_time):
        src,dst=edge_index;out=self.self_linear(x)
        # Last relation is explicit self: use the learned self path exactly once.
        for r in range(len(self.weights)):
            m=edge_type==r
            if not m.any():continue
            degree=torch.bincount(dst[m],minlength=len(x)).clamp_min(1)
            out.index_add_(0,dst[m],(x[src[m]]@self.weights[r])/degree[dst[m],None])
        return F.relu(out)


class HGTModel(nn.Module):
    def __init__(self,inputs,width,types,relations,classes,heads=8,layers=3,dropout=.2,arm='hgt'):
        super().__init__();self.arm=arm
        self.adapters=nn.ModuleList([nn.Linear(inputs,width) for _ in range(types)])
        self.drop=nn.Dropout(dropout)
        self.layers=nn.ModuleList([HGTLayer(width,types,relations,heads,dropout,use_rte=arm!='hgt_no_rte') if arm!='rgcn' else RGCNLayer(width,relations) for _ in range(layers)])
        self.classifier=nn.Linear(width,classes)
    def forward(self,batch):
        x,nt,ei,er,et,targets,labels=batch
        z=x.new_zeros((len(x),self.classifier.in_features))
        for t,adapter in enumerate(self.adapters):
            mask=nt==t;z[mask]=torch.tanh(adapter(x[mask]))
        z=self.drop(z)
        for layer in self.layers:z=layer(z,nt,ei,er,et)
        return self.classifier(z[targets]).log_softmax(-1)


class OAGGraph:
    """Compatibility container for the authors' released dill Graph object."""
    def get_types(self):return list(self.node_feature)
    def get_meta_graph(self):
        return [(t,s,r) for t,sv in self.edge_list.items() for s,rv in sv.items() for r in rv]


def inert_legacy_code(*args):
    # Python3.7 bytecode in old defaultdict factories is not executed.
    return None


def inert_legacy_function(*args):
    # The downloaded graph stores populated maps. Their historical lambda
    # factories are irrelevant to read-only sampling; missing keys must not be used.
    return dict


def legacy_type(name):
    return inert_legacy_code if name=='CodeType' else dill._dill._load_type(name)


class OAGUnpickler(dill.Unpickler):
    def find_class(self,module,name):
        if name=='Graph' and module in ('pyHGT.data','GPT_GNN.data','data'):return OAGGraph
        if module=='pandas.core.indexes.numeric':return pd.Index
        if module in ('data','pyHGT.data','GPT_GNN.data') and name=='__dict__':return {}
        if module=='dill._dill' and name=='_load_type':return legacy_type
        if module=='dill._dill' and name=='_create_function':return inert_legacy_function
        return super().find_class(module,name)


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_oag(path,expected_sha256):
    """Verify trusted release bytes before deserializing; missing hash fails closed."""
    if not expected_sha256 or file_sha256(path)!=expected_sha256:raise ValueError('OAG data SHA256 mismatch or missing expected hash')
    with open(path,'rb') as f:g=OAGUnpickler(f).load()
    assert 'paper' in g.node_feature and 'field' in g.node_feature
    return g


def field_protocol(graph):
    """L2 task: candidate order and time boundaries follow train_paper_field.py."""
    candidates=list(graph.edge_list['field']['paper']['PF_in_L2'])
    pairs={'train':{},'valid':{},'test':{}}
    for paper,fields in graph.edge_list['paper']['field']['rev_PF_in_L2'].items():
        for field,year in fields.items():
            if year is None:raise ValueError('Untimed target label requires an explicit protocol decision')
            split='train' if year<2015 else ('valid' if year<=2016 else 'test')
            if paper not in pairs[split]:pairs[split][paper]=[[],year]
            pairs[split][paper][0].append(field)
    # Release shuffles complete train/validation dictionaries with seed43.
    rng=np.random.RandomState(43)
    for split in ('train','valid'):
        ids=rng.choice(list(pairs[split]),len(pairs[split]),replace=False)
        pairs[split]={int(i):pairs[split][i] for i in ids}
    ids=[set(p) for p in pairs.values()]
    assert not any(ids[i]&ids[j] for i in range(3) for j in range(i+1,3))
    return candidates,pairs


def sample_oag(graph,pairs,candidates,seed,batch_size=256,depth=6,width=128,cutoff=2014):
    """Release HGSampling, serial RNG, degree budget, time inheritance and induced edges.
    Adapted from MIT pyHGT data.py. Cutoff filters candidate expansion; the release
    does NOT re-filter induced edge timestamps. This is disclosed, not silently fixed.
    """
    rng=np.random.RandomState(seed)
    ids=rng.choice(list(pairs),batch_size,replace=False)
    selected=defaultdict(dict);budget=defaultdict(dict)
    def add_budget(t,target,year):
        for s,relations in graph.edge_list[t].items():
            for r,adj in relations.items():
                if r=='self' or target not in adj:continue
                neighbors=adj[target]
                choices=list(neighbors) if len(neighbors)<width else rng.choice(list(neighbors),width,replace=False)
                for node in choices:
                    when=neighbors[node] if neighbors[node] is not None else year
                    if when>cutoff or node in selected[s]:continue
                    old=budget[s].get(node,[0.,0])
                    budget[s][node]=[old[0]+1/len(choices),when]
    for i in ids:selected['paper'][int(i)]=[len(selected['paper']),pairs[i][1]]
    for i in ids:add_budget('paper',i,pairs[i][1])
    for _ in range(depth):
        for t in list(budget):
            keys=np.array(list(budget[t]));n=len(keys)
            if n==0:continue
            if width>n:chosen=np.arange(n)
            else:
                probability=np.array([v[0] for v in budget[t].values()])**2;probability/=probability.sum()
                chosen=rng.choice(n,width,p=probability,replace=False)
            nodes=keys[chosen]
            for node in nodes:selected[t][node]=[len(selected[t]),budget[t][node][1]]
            for node in nodes:
                add_budget(t,node,budget[t][node][1]);budget[t].pop(node)
    types=graph.get_types();offset={};features=[];node_types=[];times=[];node_ids={};total=0
    for t_id,t in enumerate(types):
        offset[t]=total;keys=list(selected[t]);node_ids[t]=keys;total+=len(keys)
        if not keys:continue
        frame=graph.node_feature[t].loc[keys]
        emb=np.asarray(list(frame['node_emb'].to_numpy()),dtype=float) if 'node_emb' in frame else np.zeros((len(keys),400))
        feat=np.concatenate([emb,np.asarray(list(frame['emb'].to_numpy()),dtype=float),np.log10(frame['citation'].to_numpy(dtype=float)[:,None]+.01)],axis=1)
        features.append(feat);node_types.extend([t_id]*len(keys));times.extend([v[1] for v in selected[t].values()])
    # Same relation-name indexing as the release; names must identify unique meta edges.
    meta=graph.get_meta_graph();relation={r:i for i,(_,_,r) in enumerate(meta)};relation['self']=len(relation)
    if len(relation)!=len(meta)+1:raise ValueError('Repeated relation name: audit meta-relation indexing before execution')
    edges=[];rels=[]
    for t in types:
        for node,(local,_) in selected[t].items():edges.append([offset[t]+local,offset[t]+local]);rels.append(relation['self'])
    for t,srcs in graph.edge_list.items():
        for s,relations in srcs.items():
            for r,adj in relations.items():
                for target,(tid,_) in selected[t].items():
                    if target not in adj:continue
                    for source in adj[target]:
                        if source not in selected[s]:continue
                        sid=selected[s][source][0]
                        # Remove the task label edge in BOTH directions for all seed papers.
                        if r=='rev_PF_in_L2' and t=='paper' and tid<batch_size:continue
                        if r=='PF_in_L2' and s=='paper' and sid<batch_size:continue
                        edges.append([offset[s]+sid,offset[t]+tid]);rels.append(relation[r])
    ei=torch.tensor(edges,dtype=torch.long).T;ts=torch.tensor(times)
    et=(ts[ei[1]]-ts[ei[0]]+120).long()
    y=np.zeros((len(ids),len(candidates)),dtype=np.float32);ci={v:i for i,v in enumerate(candidates)}
    for row,node in enumerate(ids):
        for label in pairs[node][0]:y[row,ci[label]]=1
    y/=y.sum(1,keepdims=True)
    batch=(torch.tensor(np.concatenate(features),dtype=torch.float32),torch.tensor(node_types),ei,torch.tensor(rels),et,torch.arange(batch_size)+offset['paper'],torch.from_numpy(y))
    audit={'seed':int(seed),'paper_ids':ids.tolist(),'nodes':total,'edges':len(edges),'node_ids':{k:[int(i) for i in v] for k,v in node_ids.items()},'relation_ids':relation}
    return batch,audit


def ranking_metrics(log_prob,labels):
    """Release DCG method0: ranks1 AND2 have denominator1; MRR first relevant."""
    order=log_prob.argsort(descending=True,dim=1)
    relevance=labels.gather(1,order)
    discount=torch.ones(labels.shape[1],device=labels.device)
    if len(discount)>1:discount[1:]=torch.log2(torch.arange(2,len(discount)+1,device=labels.device))
    dcg=(relevance/discount).sum(1);ideal=(labels.sort(descending=True,dim=1).values/discount).sum(1)
    ndcg=dcg/ideal.clamp_min(1e-12)
    relevant=relevance>0;first=relevant.to(torch.int64).argmax(1)+1
    mrr=torch.where(relevant.any(1),1/first.float(),0.)
    return ndcg,mrr


def protocol_config(preset):
    common={'heads':8,'dropout':.2,'lr':.001,'weight_decay':.01,'clip':.25,'selection':'valid_ndcg','device':'cpu'}
    if preset=='paper':common.update(width=256,layers=3,epochs=200,batches=32,repeat=2,batch_size=256,depth=6,sample_width=128,test_batches=10,selection='valid_loss')
    elif preset=='release':common.update(width=400,layers=4,epochs=200,batches=32,repeat=2,batch_size=256,depth=6,sample_width=128,test_batches=10)
    elif preset=='teaching':common.update(width=32,layers=2,epochs=10,batches=4,repeat=1,batch_size=32,depth=2,sample_width=16,test_batches=10)
    elif preset=='smoke':common.update(width=32,layers=2,epochs=1,batches=1,repeat=1,batch_size=8,depth=2,sample_width=8,test_batches=1)
    else:raise ValueError(preset)
    return common


def train_oag(graph,seed,config,arm='hgt',progress=False):
    """Full visible serial trainer; fixed declared streams shared across comparison arms.
    Fresh seeds and serial sampling differ from historical multiprocessing streams.
    Test scores never select a checkpoint. Retain per-query IDs and ranking metrics.
    """
    torch.manual_seed(seed);np.random.seed(seed);start=time.time();device=torch.device(config['device'])
    candidates,pairs=field_protocol(graph);types=graph.get_types();relations=len(graph.get_meta_graph())+1
    input_dim=len(graph.node_feature['paper']['emb'].iloc[0])+401
    model=HGTModel(input_dim,config['width'],len(types),relations,len(candidates),config['heads'],config['layers'],config['dropout'],arm).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1000,eta_min=1e-6)
    best=-math.inf;state=None;trace=[];step=1500
    streams=np.random.RandomState(seed+10000)
    def make(split,number):
        cutoff=2014 if split=='train' else (2016 if split=='valid' else max(t for t in graph.times if t is not None))
        b,a=sample_oag(graph,pairs[split],candidates,number,config['batch_size'],config['depth'],config['sample_width'],cutoff)
        return tuple(t.to(device) for t in b),a
    for epoch in range(1,config['epochs']+1):
        seeds=[int(streams.randint(2**32-1)) for _ in range(config['batches']+1)]
        model.train();losses=[]
        for repeat in range(config['repeat']):
            for number in seeds[:-1]:
                b,_=make('train',number);opt.zero_grad();pred=model(b)
                loss=F.kl_div(pred,b[-1],reduction='batchmean');loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),config['clip']);opt.step();step+=1;scheduler.step(step);losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            b,a=make('valid',seeds[-1]);pred=model(b);vl=float(F.kl_div(pred,b[-1],reduction='batchmean'));nd,_=ranking_metrics(pred,b[-1]);vn=float(nd.mean())
        score=-vl if config['selection']=='valid_loss' else vn
        if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected_epoch=epoch
        trace.append({'epoch':epoch,'train_kl':float(np.mean(losses)),'valid_kl':vl,'valid_ndcg':vn,'train_sampling_seeds':seeds[:-1],'valid_sampling_seed':seeds[-1]})
        if progress:print(f'{arm} seed{seed} epoch{epoch}: train KL {np.mean(losses):.4f}; valid NDCG {vn:.4f}',flush=True)
    model.load_state_dict(state);model.eval();queries=[];test_rng=np.random.RandomState(seed+20000)
    with torch.no_grad():
        for _ in range(config['test_batches']):
            b,a=make('test',int(test_rng.randint(2**32-1)));pred=model(b);nd,mrr=ranking_metrics(pred,b[-1])
            for i,node in enumerate(a['paper_ids']):
                queries.append({'paper_id':node,'sample_seed':a['seed'],'ndcg':float(nd[i]),'mrr':float(mrr[i]),'relevant_ranks':(torch.nonzero(b[-1][i][pred[i].argsort(descending=True)]>0).flatten()+1).cpu().tolist(),'true_field_ids':[int(i) for i in pairs['test'][node][0]]})
    result={'seed':seed,'arm':arm,'config':config,'parameters':sum(p.numel() for p in model.parameters()),'selected_epoch':selected_epoch,'trace':trace,'test_queries':queries,'test_ndcg':float(np.mean([q['ndcg'] for q in queries])),'test_mrr':float(np.mean([q['mrr'] for q in queries])),'seconds':time.time()-start,'split_sizes':{k:len(v) for k,v in pairs.items()},'candidate_ids':[int(i) for i in candidates]}
    return result,model


def frequency_baseline(graph,queries):
    """Fit field frequency on training labels, evaluate fixed supplied query IDs."""
    candidates,pairs=field_protocol(graph);counts={f:0 for f in candidates}
    for fields,year in pairs['train'].values():
        for field in fields:counts[field]+=1
    order=sorted(candidates,key=lambda f:-counts[f]);ranks={f:i+1 for i,f in enumerate(order)}
    ndcg=[];mrr=[]
    for q in queries:
        rr=np.array([ranks[f] for f in q['true_field_ids']]);discount=lambda x:np.where(x==1,1.,np.log2(x))
        ndcg.append(float((1/discount(rr)).sum()/(1/discount(np.arange(1,len(rr)+1))).sum()));mrr.append(float(1/rr.min()))
    return {'ndcg':float(np.mean(ndcg)),'mrr':float(np.mean(mrr)),'top_field_id':int(order[0]),'train_prevalence':counts[order[0]]/len(pairs['train'])}

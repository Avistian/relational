"""L064: full historical default-classifier TabPFN v2 computation, visible and uncached.

Independently written against tabpfn 2.0.9. Only CPU float32/float64 numeric inputs;
all 12 layers and every checkpoint parameter are used. No training/resume claim.
"""
import hashlib,importlib.metadata,inspect,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
CHECKPOINT_SHA='f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f'
CHECKPOINT_URL='https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/f851f2a3c941544733b712d8c0f96dfae9b28862/tabpfn-v2-classifier.ckpt'
MODEL_CONFIG=dict(width=192,heads=6,hidden=768,layers=12,classes=10,group_size=2)
RECIPE=dict(temperature=.9,remove_constants=True,transform='none',fingerprint=False,feature_permutation=False,class_permutation=False,cache=False)
PRESETS={'smoke':dict(datasets=['diabetes'],seeds=[0],cap=120),'lab':dict(datasets=['diabetes'],seeds=[7],cap=None),'closer':dict(datasets=['diabetes','blood_transfusion','wdbc'],seeds=[0,1,2],cap=None)}

def group_features(x,group_size=2):
    """B×N×F -> B×N×ceil(F/g)×g, preserving row/feature identities."""
    if x.ndim!=3 or group_size<1 or x.shape[-1]<1:raise ValueError('Expected B,N,F and positive group size')
    padding=(-x.shape[-1])%group_size
    return nn.functional.pad(x,(0,padding)).reshape(*x.shape[:2],-1,group_size)

def impute_with_flags(x,n):
    """Context mean replacement; flags 0 observed, -2 NaN, +2/+4 infinities.

    The numeric wrapper rejects infinities. This helper retains source low-level
    semantics for independent encoder comparisons.
    """
    context=x[:,:n];valid=~torch.isnan(context)
    means=torch.where(valid,context,0).sum(1)/valid.sum(1).clamp_min(1)
    flags=(torch.isnan(x)*-2+torch.isposinf(x)*2+torch.isneginf(x)*4).to(x.dtype)
    filled=torch.where(torch.isfinite(x),x,means[:,None])
    return filled,flags

def soften_outliers(x,n,sigma):
    """Context two-pass bounds and logarithmic tail compression from v2 source."""
    context=x[:,:n]
    mean=context.mean(1,keepdim=True);std=context.std(1,correction=1,keepdim=True)
    clean=torch.where((context<mean-sigma*std)|(context>mean+sigma*std),float('nan'),context)
    mask=~torch.isnan(clean);count=mask.sum(1,keepdim=True)
    mean=torch.where(mask,clean,0).sum(1,keepdim=True)/count.clamp_min(1)
    std=torch.sqrt(torch.where(mask,(clean-mean).square(),0).sum(1,keepdim=True)/(count-1))
    lower=mean-sigma*std;upper=mean+sigma*std
    x=torch.maximum(-torch.log(1+x.abs())+lower,x)
    return torch.minimum(torch.log(1+x.abs())+upper,x)

def encode_groups(x,n,outlier_sigma=None):
    """Missing replacement -> context sample z-score -> source active-group scaling.

    Active values are counted across ALL supplied rows, matching the historical
    uncached encoder. This is a deliberately exposed boundary, not a guarantee
    of query independence. Flag channels are NOT active-count scaled.
    """
    filled,flags=impute_with_flags(x,n)
    if outlier_sigma is not None:filled=soften_outliers(filled,n,outlier_sigma)
    context=filled[:,:n];mean=context.mean(1,keepdim=True)
    std=context.std(1,correction=1,keepdim=True)+1e-20
    if n==1:std=torch.ones_like(std)
    z=((filled-mean)/std).clamp(-100,100)
    active=(z[:,1:]!=z[:,:1]).any(1).sum(-1,keepdim=True).clamp_min(1)
    z=z*torch.sqrt(x.shape[-1]/active[:,None])
    return torch.cat([z,flags],-1)

def encode_targets(y,total_rows):
    """B×C -> B×N×2: ordinal target and missing flag; query labels unavailable."""
    if y.ndim!=2 or not 1<=y.shape[1]<total_rows or not torch.isfinite(y).all():raise ValueError('Context labels only, with at least one query')
    missing=torch.full((len(y),total_rows-y.shape[1]),float('nan'),dtype=y.dtype,device=y.device)
    values,flags=impute_with_flags(torch.cat([y,missing],1).unsqueeze(-1),y.shape[1])
    ranks=torch.stack([(values[b,...,0,None]>torch.unique(y[b])).sum(-1) for b in range(len(y))]).to(y.dtype)
    return torch.stack([ranks,flags[...,0]],-1)

def attention_mix(q,k,v):
    """...×H×receivers×d -> weighted values; K/V may broadcast one head."""
    if q.shape[-1]!=k.shape[-1] or k.shape[-2]!=v.shape[-2]:raise ValueError('Attention shape mismatch')
    # Historical CPU branch rounds the inverse-square-root scalar to float32.
    scale=torch.sqrt(torch.tensor(1.0/q.shape[-1])).to(q.device)
    return ((q@k.transpose(-2,-1))*scale).softmax(-1)@v

class PackedAttention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.width=width
        self.qkv=nn.Parameter(torch.empty(3,heads,width//heads,width))
        self.out=nn.Parameter(torch.empty(heads,width//heads,width))
        nn.init.normal_(self.qkv,std=.02);nn.init.normal_(self.out,std=.02)
    def forward(self,receivers,senders=None,first_kv=False):
        senders=receivers if senders is None else senders
        q=torch.einsum('...sd,hkd->...hsk',receivers,self.qkv[0])
        weights=self.qkv[1:,:1] if first_kv else self.qkv[1:]
        k,v=torch.einsum('...sd,jhkd->j...hsk',senders,weights)
        mixed=attention_mix(q,k,v)
        return torch.einsum('...hsk,hkd->...sd',mixed,self.out)

def row_attention(h,n,attention):
    """B,N,G,D -> B,N,G,D; context MHA, query first-head-KV MQA."""
    if h.ndim!=4 or not 1<=n<h.shape[1]:raise ValueError('Context/query row split')
    rows=h.transpose(1,2)
    context=rows[:,:,:n];query=rows[:,:,n:]
    context_update=attention(context,context)
    query_update=attention(query,context,first_kv=True)
    return torch.cat([context_update,query_update],2).transpose(1,2)

def postnorm_update(h,update):
    """Every sublayer adds its input BEFORE non-affine LayerNorm(eps=1e-5)."""
    return nn.functional.layer_norm(h+update,(h.shape[-1],),eps=1e-5)

class V2Block(nn.Module):
    def __init__(self,width=192,heads=6,hidden=768):
        super().__init__();self.feature=PackedAttention(width,heads);self.row=PackedAttention(width,heads)
        self.ff1=nn.Linear(width,hidden,bias=False);self.ff2=nn.Linear(hidden,width,bias=False)
    def forward(self,h,n):
        h=postnorm_update(h,self.feature(h))
        h=postnorm_update(h,row_attention(h,n,self.row))
        return postnorm_update(h,self.ff2(nn.functional.gelu(self.ff1(h))))

class TabPFNv2(nn.Module):
    def __init__(self,width=192,heads=6,hidden=768,layers=12,classes=10,group_size=2):
        super().__init__();self.config=dict(width=width,heads=heads,hidden=hidden,layers=layers,classes=classes,group_size=group_size)
        self.x_encoder=nn.Linear(2*group_size,width,bias=False);self.y_encoder=nn.Linear(2,width)
        self.position=nn.Linear(width//4,width)
        self.blocks=nn.ModuleList([V2Block(width,heads,hidden) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,classes))
    def forward(self,x,y,seed=0,return_hidden=False,outlier_sigma=None):
        if x.ndim!=3 or y.ndim!=2 or x.shape[0]!=y.shape[0]:raise ValueError('x:B,N,F; y:B,C')
        n=y.shape[1];groups=group_features(x,self.config['group_size'])
        features=self.x_encoder(encode_groups(groups,n,outlier_sigma))
        generator=torch.Generator(device=x.device).manual_seed(seed)
        random_ids=torch.randn((groups.shape[2],self.config['width']//4),generator=generator,device=x.device,dtype=x.dtype)
        features=features+self.position(random_ids)[None,None]
        target=self.y_encoder(encode_targets(y.to(x.dtype),x.shape[1]))
        h=torch.cat([features,target.unsqueeze(2)],2)
        for block in self.blocks:h=block(h,n)
        logits=self.head(h[:,n:,-1])
        return (logits,h) if return_hidden else logits

def load_pretrained(path,model=None):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Unexpected v2 checkpoint bytes')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=TabPFNv2(**MODEL_CONFIG) if model is None else model
    mapped={}
    for key,value in saved['state_dict'].items():
        key=key.replace('encoder.5.layer.','x_encoder.').replace('y_encoder.2.layer.','y_encoder.').replace('feature_positional_embedding_embeddings.','position.').replace('decoder_dict.standard.','head.')
        key=key.replace('transformer_encoder.layers.','blocks.').replace('self_attn_between_features._w_','feature.').replace('self_attn_between_items._w_','row.').replace('mlp.linear1.','ff1.').replace('mlp.linear2.','ff2.')
        mapped[key]=value
    model.load_state_dict(mapped,strict=True);model.eval()
    return model,saved['config']

def ensure_checkpoint(root):
    import urllib.request
    path=Path(root)/'data/cache/foundation/tabpfn-v2.ckpt';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        partial=path.with_suffix('.partial');urllib.request.urlretrieve(CHECKPOINT_URL,partial)
        if hashlib.sha256(partial.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Downloaded checkpoint checksum')
        partial.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Cached checkpoint checksum')
    return path

def predict_numeric(model,context_x,context_y,query_x,seed=0,temperature=.9):
    """Complete deliberately simple released configuration, with raw numeric views.

    Includes context-fitted constant removal, label encoding and probability
    normalization. One uncached call; no batch/resume equivalence is assumed.
    """
    cx=np.asarray(context_x,dtype=np.float32);qx=np.asarray(query_x,dtype=np.float32)
    if cx.ndim!=2 or qx.ndim!=2 or cx.shape[1]!=qx.shape[1] or np.isinf(cx).any() or np.isinf(qx).any():raise ValueError('Numeric matrix, matching columns, no infinity')
    classes,y=np.unique(context_y,return_inverse=True)
    if not 2<=len(classes)<=10 or len(y)!=len(cx) or temperature<=0:raise ValueError('2..10 context classes and positive temperature')
    keep=(cx==cx[:1]).mean(0)<1
    if not keep.any():raise ValueError('All features are constant')
    x=torch.from_numpy(np.concatenate([cx[:,keep],qx[:,keep]]))[None]
    labels=torch.tensor(y,dtype=x.dtype)[None]
    model.eval()
    with torch.no_grad():logits=model(x,labels,seed=seed)[0,:,:len(classes)]
    p=(logits/temperature).softmax(-1);p=p/p.sum(-1,keepdim=True)
    return p,dict(classes=classes.tolist(),kept_columns=np.where(keep)[0].tolist(),logits=logits.tolist(),seed=seed,temperature=temperature,groups=(int(keep.sum())+1)//2)

def load_dataset(name):
    """All rows, numeric paper datasets. wdbc is sklearn's same UCI source (569×30)."""
    if name=='wdbc':
        from sklearn.datasets import load_breast_cancer
        data=load_breast_cancer();x=data.data;y=data.target
    else:
        from relkit.data import load_tier_a
        x,y=load_tier_a(name);x=np.asarray(x,dtype=np.float32);y=np.asarray(y)
    _,y=np.unique(y,return_inverse=True)
    return np.asarray(x,dtype=np.float32),y

def run_experiment(root,model,config=None):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss,roc_auc_score
    config=dict(PRESETS['lab'] if config is None else config);records=[]
    for dataset in config['datasets']:
        x,y=load_dataset(dataset);ids=np.arange(len(y))
        for seed in config['seeds']:
            selected=ids
            if config['cap'] and config['cap']<len(ids):selected,_=train_test_split(ids,train_size=config['cap'],stratify=y,random_state=seed+6400)
            tr,te=train_test_split(selected,test_size=.5,stratify=y[selected],random_state=seed)
            # Compare a legal context-label intervention with fixed weights/features.
            for condition in ['observed','shuffled']:
                labels=y[tr] if condition=='observed' else y[tr][np.random.default_rng(seed+64000).permutation(len(tr))]
                start=time.perf_counter();p,trace=predict_numeric(model,x[tr],labels,x[te],seed=seed,temperature=RECIPE['temperature']);seconds=time.perf_counter()-start
                probabilities=p.numpy();auc=roc_auc_score(y[te],probabilities[:,1])
                records.append(dict(dataset=dataset,seed=seed,condition=condition,train_ids=tr.tolist(),test_ids=te.tolist(),context_labels=labels.tolist(),targets=y[te].tolist(),probabilities=probabilities.tolist(),trace=trace,log_loss=log_loss(y[te],probabilities),auc=auc,seconds=seconds))
                print(dataset,seed,condition,records[-1]['log_loss'],flush=True)
    return dict(config=config,recipe=RECIPE,checkpoint_sha256=CHECKPOINT_SHA,records=records,verdict='INCOMPARABLE',pretraining='NOT_RUN',paper_benchmark='NOT_RUN')

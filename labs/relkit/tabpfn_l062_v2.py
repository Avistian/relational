"""L062: visible historical TabPFN v1 numeric inference, independently weight-checked.

TabPFN 0.1.11 is a reference only. Core inference below needs torch/numpy/sklearn.
Original checkpoint is hash checked before deserialization. No pretraining or resume.
"""
import hashlib,inspect,importlib.metadata,itertools,json,math,random,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import PowerTransformer

CHECKPOINT_SHA='3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392'
CHECKPOINT_URL='https://raw.githubusercontent.com/PriorLabs/TabPFN/v1.0.0/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'
MODEL_CONFIG=dict(features=100,width=512,heads=4,hidden=1024,layers=12,classes=10)
PRESETS={'smoke':dict(datasets=['diabetes'],seeds=[0],cap=120,views=[1,4]),
         'lab':dict(datasets=['diabetes','blood_transfusion','wdbc'],seeds=[0,1,2],cap=None,views=[1,4]),
         'closer':dict(datasets=['diabetes','blood_transfusion','wdbc'],seeds=[0,1,2,3,4],cap=None,views=[1,32])}

def masked_moments(x,mask):
    count=mask.sum(0);mean=torch.where(mask,x,0).sum(0)/count
    std=(torch.where(mask,(x-mean).square(),0).sum(0)/(count-1)).sqrt()
    return mean,std

def normalize_context(x,n):
    """N×F, context-fitted sample std; same 1e-6 stabilization and ±100 clamp as v1."""
    if x.ndim!=2 or not 2<=n<len(x):raise ValueError('Need N×F and at least two context rows plus queries')
    mean,std=masked_moments(x[:n],~torch.isnan(x[:n]))
    return ((x-mean)/(std+1e-6)).clamp(-100,100)

def soften_outliers(x,n,n_sigma=4):
    """Two context moment passes; logarithmic tail compression, not hard clipping."""
    c=x[:n];mean,std=masked_moments(c,~torch.isnan(c))
    mask=(c>=mean-n_sigma*std)&(c<=mean+n_sigma*std)&~torch.isnan(c)
    mean,std=masked_moments(c,mask);lo,hi=mean-n_sigma*std,mean+n_sigma*std
    x=torch.maximum(-torch.log1p(x.abs())+lo,x)
    return torch.minimum(torch.log1p(x.abs())+hi,x)

def scale_pad(x,max_features=100):
    """Flexible encoder: K/k magnitude factor then zero padding; not sqrt(K/k)."""
    k=x.shape[-1]
    if not 1<=k<=max_features:raise ValueError('Need 1..K active features')
    return nn.functional.pad(x/(k/max_features),(0,max_features-k))

def preprocess_numeric(x,n,transform='none'):
    """Numeric v1 wrapper subset: context moments, constants, power, outliers.
    Missing input values are retained until the model's explicit nan_to_num.
    No categorical indices, ranking, feature subsampling or query-fitted moments.
    """
    if transform not in ['none','power_all']:raise ValueError('Unsupported transform')
    z=normalize_context(x,n)
    keep=torch.tensor([len(torch.unique(z[:n,j]))>1 for j in range(z.shape[1])],device=z.device)
    z=z[:,keep]
    if z.shape[-1]==0:raise ValueError('All features are constant in context')
    if transform=='power_all':
        a=z.detach().cpu().numpy().copy()
        for j in range(a.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                try:
                    pt=PowerTransformer(standardize=True).fit(a[:n,j:j+1])
                    a[:,j:j+1]=pt.transform(a[:,j:j+1])
                except (ValueError,RuntimeWarning,UserWarning,FloatingPointError):pass
        z=torch.tensor(a,dtype=x.dtype,device=x.device)
    return soften_outliers(z,n),keep

def attention_mix(q,k,v):
    """B×H×receivers×headwidth; only context keys/values are supplied."""
    return torch.softmax(q@k.transpose(-2,-1)/math.sqrt(q.shape[-1]),dim=-1)@v

def postnorm_update(h,attention_output,block):
    """Attention residual/norm then GELU FFN residual/norm, dropout=0."""
    h=block.norm1(h+attention_output)
    return block.norm2(h+block.ff2(nn.functional.gelu(block.ff1(h))))

class V1Block(nn.Module):
    def __init__(self,width,heads,hidden):
        super().__init__();self.width=width;self.heads=heads
        if width%heads:raise ValueError('heads must divide width')
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.ff1=nn.Linear(width,hidden);self.ff2=nn.Linear(hidden,width)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        nn.init.xavier_uniform_(self.qkv.weight);nn.init.zeros_(self.qkv.bias)
        for layer in [self.out,self.ff2]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,h,n):
        b,t,d=h.shape
        q,k,v=self.qkv(h).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        # Rectangular attention computes n*n + query*n scores; no query self edge.
        mixed=attention_mix(q,k[:,:,:n],v[:,:,:n])
        mixed=mixed.transpose(1,2).reshape(b,t,d)
        return postnorm_update(h,self.out(mixed),self)

class TabPFNv1(nn.Module):
    def __init__(self,features=100,width=512,heads=4,hidden=1024,layers=12,classes=10):
        super().__init__();self.config=dict(features=features,width=width,heads=heads,hidden=hidden,layers=layers,classes=classes)
        self.x_encoder=nn.Linear(features,width);self.y_encoder=nn.Linear(1,width)
        self.blocks=nn.ModuleList([V1Block(width,heads,hidden) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,classes))
    def forward(self,x,y):
        # x:B×(C+Q)×100, y:B×C. Query labels cannot enter this signature.
        n=y.shape[1]
        if x.ndim!=3 or y.ndim!=2 or x.shape[0]!=y.shape[0] or not 1<=n<x.shape[1]:raise ValueError('Bad context/query shape')
        h=self.x_encoder(torch.nan_to_num(x))
        h=torch.cat([h[:,:n]+self.y_encoder(y.to(x.dtype).unsqueeze(-1)),h[:,n:]],dim=1)
        for block in self.blocks:h=block(h,n)
        return self.head(h[:,n:])

def load_pretrained(path,model=None):
    """Strict complete weight mapping; criterion buffer is the only non-model entry."""
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Unexpected checkpoint bytes')
    state,_,config=torch.load(path,map_location='cpu',weights_only=False)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model=TabPFNv1(**MODEL_CONFIG) if model is None else model
    mapped={}
    for key,value in state.items():
        if key=='criterion.weight':continue
        new=key.replace('transformer_encoder.layers.','blocks.').replace('self_attn.in_proj_weight','qkv.weight').replace('self_attn.in_proj_bias','qkv.bias').replace('self_attn.out_proj.','out.').replace('linear1.','ff1.').replace('linear2.','ff2.')
        if new.startswith('encoder.'):new='x_encoder.'+new[len('encoder.'):]
        if new.startswith('decoder.'):new='head.'+new[len('decoder.'):]
        mapped[new]=value
    model.load_state_dict(mapped,strict=True);model.eval()
    return model,config

def view_configurations(features,classes,views,seed=0):
    """Match historical torch+Python ordering; none/power alternate per rotation."""
    if views<1:raise ValueError('At least one view')
    generator=torch.Generator().manual_seed(seed)
    fs=torch.randperm(features,generator=generator).tolist()
    cs=torch.randperm(classes,generator=generator).tolist()
    pairs=list(itertools.product(cs,fs));random.Random(seed).shuffle(pairs)
    return [(c,f,t) for (c,f),t in itertools.product(pairs,['none','power_all'])][:views]

def aggregate_views(logits,shifts,classes,temperature=.8):
    """V×Q×10 -> Q×K: undo each class rotation, mean logits, then softmax."""
    if logits.ndim!=3 or len(shifts)!=len(logits) or not 2<=classes<=logits.shape[-1] or temperature<=0:raise ValueError('Invalid ensemble')
    aligned=[torch.cat([z[:,shift:classes],z[:,:shift]],-1) for z,shift in zip(logits,shifts)]
    return (torch.stack(aligned).mean(0)/temperature).softmax(-1)

def predict_numeric(model,context_x,context_y,query_x,views=4,seed=0,query_batch=256):
    """Full numeric inference recipe using the learner's model and five live tasks."""
    cx=torch.as_tensor(context_x,dtype=torch.float32);qx=torch.as_tensor(query_x,dtype=torch.float32)
    y=torch.as_tensor(context_y,dtype=torch.long);classes=len(torch.unique(y))
    if not torch.equal(torch.unique(y),torch.arange(classes)) or not 2<=classes<=10:raise ValueError('Encode context classes as contiguous 0..K-1')
    if cx.shape[1]!=qx.shape[1] or cx.shape[1]>100:raise ValueError('Feature dimensions')
    configs=view_configurations(cx.shape[1],classes,views,seed);n=len(cx)
    transformed={t:preprocess_numeric(torch.cat([cx,qx]),n,t)[0] for t in {c[2] for c in configs}}
    outputs=[];model.eval()
    with torch.no_grad():
        for shift,feature_shift,transform in configs:
            z=transformed[transform]
            # Source draws shifts before constant-feature removal; slicing preserves its behavior.
            z=torch.cat([z[:,feature_shift:],z[:,:feature_shift]],-1)
            z=scale_pad(z,model.config['features']);yc=((y+shift)%classes).float()[None]
            chunks=[]
            for query in z[n:].split(query_batch):chunks.append(model(torch.cat([z[:n],query])[None],yc)[0])
            outputs.append(torch.cat(chunks))
    logits=torch.stack(outputs)
    return aggregate_views(logits,[c[0] for c in configs],classes),{'configurations':configs,'logits':logits.tolist(),'temperature':.8}

def ensure_checkpoint(root):
    import urllib.request
    path=Path(root)/'data/cache/foundation/v1/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True);urllib.request.urlretrieve(CHECKPOINT_URL,path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Checkpoint hash mismatch')
    return path

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

def run_experiment(root,preset='lab',model=None,config=None):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss,roc_auc_score
    cfg=dict(PRESETS[preset] if config is None else config)
    model,_=load_pretrained(ensure_checkpoint(root)) if model is None else (model,None)
    result=dict(preset=preset,config=cfg,checkpoint_sha256=CHECKPOINT_SHA,records=[],datasets={},status='INCOMPARABLE',scope='Pretrained inference through visible model; no new pretraining; paper dataset subset and local split seeds')
    for name in cfg['datasets']:
        x,y=load_dataset(name)
        result['datasets'][name]={'shape':list(x.shape),'x_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'y_sha256':hashlib.sha256(y.tobytes()).hexdigest()}
        for seed in cfg['seeds']:
            ids=np.arange(len(y))
            if cfg['cap'] is not None:ids=train_test_split(ids,train_size=cfg['cap'],stratify=y,random_state=620)[0]
            tr,te=train_test_split(ids,test_size=.5,stratify=y[ids],random_state=seed)
            for views in cfg['views']:
                begin=time.perf_counter();p,trace=predict_numeric(model,x[tr],y[tr],x[te],views=views,seed=seed)
                result['records'].append(dict(dataset=name,seed=seed,views=views,arm='pretrained',log_loss=log_loss(y[te],p.numpy()),auc=roc_auc_score(y[te],p[:,1].numpy()),seconds=time.perf_counter()-begin,train_ids=tr.tolist(),test_ids=te.tolist(),targets=y[te].tolist(),probabilities=p.tolist(),trace=trace))
                print(name,seed,views,result['records'][-1]['log_loss'],flush=True)
    result['versions']={p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn','scipy']}
    result['operator_reference_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result

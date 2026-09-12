"""Complete TabICL numeric inference, independently written from pinned 0.1.4.

All pretrained layers; finite numeric arrays, one view, 2–10 classes. No training,
class hierarchy, power-transform ensemble, GPU offload or inference-manager claim.
"""
import hashlib,importlib.metadata,inspect,json,math,time,types,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss

CHECKPOINT_SHA='f5bae1d31181a1bb4ab8e97d2a5e62a504e856f23b4ea62c7fcc2f8eec4995b6'
CHECKPOINT_URL='https://huggingface.co/jingang/TabICL-clf/resolve/eaf789a9b25ee8486d6f48997ba076f850bbc30b/tabicl-classifier-v1-0208.ckpt'
CHECKPOINT_FILE='tabicl-v1-0208.ckpt'
MODEL_CONFIG=dict(max_classes=10,embed_dim=128,col_num_blocks=3,col_nhead=4,col_num_inds=128,row_num_blocks=3,row_nhead=8,row_num_cls=4,row_rope_base=100000,icl_num_blocks=12,icl_nhead=4,ff_factor=2,dropout=0.,activation='gelu',norm_first=True)
PROTOCOL=dict(temperature=.9,normalization='none',views=1,query_fraction=.2,context_fractions=[.125,.375,1.],float='float32',outlier_threshold=4.,query_keys=False,embedding_with_query=False)
PRESETS66={'smoke':dict(datasets=['diabetes'],seeds=[7]),'lab':dict(datasets=['diabetes'],seeds=[7]),'closer':dict(datasets=['diabetes','blood_transfusion','wdbc'],seeds=[7,17,27])}

def attention_mix(q,k,v):
    """Actual per-head attention; final axes are sequence, head coordinates."""
    if q.shape[-1]!=k.shape[-1] or k.shape[-2]!=v.shape[-2] or k.shape[-2]==0:raise ValueError('Incompatible or empty attention memory')
    scores=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
    return scores.softmax(-1)@v

def context_keys(k,v,n_context):
    """Every dataset-stage reader sees only the context prefix as keys/values."""
    if not 0<n_context<=k.shape[-2] or k.shape!=v.shape:raise ValueError('Invalid context boundary')
    return k[...,:n_context,:],v[...,:n_context,:]

def rotary_pairs(x,freqs):
    """Rotate adjacent head-coordinate pairs; positions include the four CLSs."""
    if x.shape[-1]!=2*len(freqs):raise ValueError('One frequency per adjacent coordinate pair')
    angle=torch.arange(x.shape[-2],device=x.device,dtype=freqs.dtype)[:,None]*freqs[None,:]
    angle=angle.repeat_interleave(2,-1)
    pairs=x.reshape(*x.shape[:-1],-1,2)
    perpendicular=torch.stack([-pairs[...,1],pairs[...,0]],-1).flatten(-2)
    return (x*angle.cos()+perpendicular*angle.sin()).to(x.dtype)

def inducing_memory(src,inducing,n_context,read_block,write_block):
    """Two complete MABs. Learned readers -> context memory -> all cell readers."""
    if not 0<n_context<=src.shape[-2]:raise ValueError('Invalid column context')
    readers=inducing.expand(*src.shape[:-2],*inducing.shape)
    memory=read_block(readers,src[...,:n_context,:])
    return write_block(src,memory)

def conditional_affine(values,hidden,weight_head,bias_head,weight_norm,bias_norm):
    """Normalize generated scale and offset separately before applying the scalar."""
    weights=weight_norm(weight_head(hidden));biases=bias_norm(bias_head(hidden))
    return values*weights+biases

class PackedAttention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads
        self.in_proj_weight=nn.Parameter(torch.empty(3*width,width));self.in_proj_bias=nn.Parameter(torch.empty(3*width))
        self.out_proj=nn.Linear(width,width)
    def forward(self,q,k,rope=None,n_context=None):
        # Separate projections make the Q/K/V computation visible; same packed weights.
        w=self.in_proj_weight.chunk(3);b=self.in_proj_bias.chunk(3);d=q.shape[-1];h=self.heads
        projected=[nn.functional.linear(z,wi,bi).unflatten(-1,(h,d//h)).transpose(-3,-2) for z,wi,bi in zip((q,k,k),w,b)]
        qq,kk,vv=projected
        if rope is not None:qq=rotary_pairs(qq,rope.freqs);kk=rotary_pairs(kk,rope.freqs)
        if n_context is not None:kk,vv=context_keys(kk,vv,n_context)
        out=attention_mix(qq,kk,vv).transpose(-3,-2).flatten(-2)
        return self.out_proj(out)

class AttentionBlock(nn.Module):
    def __init__(self,width,heads,ff_factor=2):
        super().__init__();self.attn=PackedAttention(width,heads)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.linear1=nn.Linear(width,width*ff_factor);self.linear2=nn.Linear(width*ff_factor,width)
    def forward(self,q,k=None,rope=None,n_context=None):
        k=q if k is None else k
        x=q+self.attn(self.norm1(q),self.norm1(k),rope,n_context)
        return x+self.linear2(nn.functional.gelu(self.linear1(self.norm2(x))))

class InducedBlock(nn.Module):
    def __init__(self,width,heads,n_inducing,ff_factor):
        super().__init__();self.ind_vectors=nn.Parameter(torch.empty(n_inducing,width))
        self.multihead_attn1=AttentionBlock(width,heads,ff_factor);self.multihead_attn2=AttentionBlock(width,heads,ff_factor)
    def forward(self,src,n_context):
        skip=(src==-100.).all(dim=(-2,-1));out=torch.full_like(src,-100.)
        if (~skip).any():out[~skip]=inducing_memory(src[~skip],self.ind_vectors,n_context,self.multihead_attn1,self.multihead_attn2)
        return out

class RotaryFrequencies(nn.Module):
    def __init__(self,head_width,base):
        super().__init__();self.freqs=nn.Parameter(1./(base**(torch.arange(0,head_width,2).float()/head_width)),requires_grad=False)

class Encoder(nn.Module):
    def __init__(self,width,heads,depth,ff_factor,rope_base=None,n_inducing=None):
        super().__init__()
        self.blocks=nn.ModuleList([AttentionBlock(width,heads,ff_factor) if n_inducing is None else InducedBlock(width,heads,n_inducing,ff_factor) for _ in range(depth)])
        self.rope=RotaryFrequencies(width//heads,rope_base) if rope_base else None
    def forward(self,x,n_context=None):
        for block in self.blocks:
            x=block(x,n_context) if isinstance(block,InducedBlock) else block(x,rope=self.rope,n_context=n_context)
        return x

class SkipLinear(nn.Linear):
    def forward(self,x):
        out=nn.functional.linear(x,self.weight,self.bias)
        return torch.where((x==-100.).all(-1,keepdim=True),-100.,out)

class ColumnEmbedding(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg['embed_dim'];self.in_linear=SkipLinear(1,d)
        self.tf_col=Encoder(d,cfg['col_nhead'],cfg['col_num_blocks'],cfg['ff_factor'],n_inducing=cfg['col_num_inds'])
        self.out_w=SkipLinear(d,d);self.out_b=SkipLinear(d,d);self.ln_w=nn.LayerNorm(d);self.ln_b=nn.LayerNorm(d)
    def forward(self,x,n_context):
        values=x.transpose(1,2).unsqueeze(-1)
        h=self.tf_col(self.in_linear(values),n_context)
        return conditional_affine(values,h,self.out_w,self.out_b,self.ln_w,self.ln_b).transpose(1,2)

class RowInteraction(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg['embed_dim'];self.cls_tokens=nn.Parameter(torch.empty(cfg['row_num_cls'],d))
        self.tf_row=Encoder(d,cfg['row_nhead'],cfg['row_num_blocks'],cfg['ff_factor'],rope_base=cfg['row_rope_base']);self.out_ln=nn.LayerNorm(d)
    def forward(self,e):
        cls=self.cls_tokens.expand(*e.shape[:2],*self.cls_tokens.shape)
        tokens=torch.cat([cls,e],-2);h=self.tf_row(tokens)
        return self.out_ln(h[...,:len(self.cls_tokens),:]).flatten(-2)

class ICLPredictor(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg['embed_dim']*cfg['row_num_cls'];self.max_classes=cfg['max_classes']
        self.tf_icl=Encoder(d,cfg['icl_nhead'],cfg['icl_num_blocks'],cfg['ff_factor']);self.ln=nn.LayerNorm(d)
        self.y_encoder=nn.Linear(self.max_classes,d);self.decoder=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,self.max_classes))
    def forward(self,r,y):
        c=y.shape[1];r=r.clone();encoded=nn.functional.one_hot(y.long(),self.max_classes).to(r.dtype)
        r[:,:c]=r[:,:c]+self.y_encoder(encoded)
        return self.decoder(self.ln(self.tf_icl(r,c)))[:,c:]

class TabICL(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=dict(MODEL_CONFIG if config is None else config)
        if self.config!=MODEL_CONFIG:raise ValueError('Only the exact pinned inference architecture is supported')
        self.col_embedder=ColumnEmbedding(self.config);self.row_interactor=RowInteraction(self.config);self.icl_predictor=ICLPredictor(self.config)
    def forward(self,x,y,return_stages=False):
        if x.ndim!=3 or y.ndim!=2 or x.shape[0]!=y.shape[0] or not 0<y.shape[1]<x.shape[1]:raise ValueError('Expected B×N×F, B×C, with context and queries')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Finite numeric inputs only')
        k=len(torch.unique(y[0]));expected=torch.arange(k,device=y.device)
        if not 2<=k<=10 or any(not torch.equal(torch.unique(v).long(),expected) for v in y):raise ValueError('Every table must use contiguous classes 0..K-1, 2<=K<=10')
        e=self.col_embedder(x,y.shape[1]);r=self.row_interactor(e);out=self.icl_predictor(r,y)[...,:k]
        return dict(column=e,row=r,logits=out) if return_stages else out

def ensure_checkpoint(root):
    path=Path(root)/'data/cache/foundation'/CHECKPOINT_FILE
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True);payload=urllib.request.urlopen(CHECKPOINT_URL).read()
        if hashlib.sha256(payload).hexdigest()!=CHECKPOINT_SHA:raise RuntimeError('Checkpoint download checksum mismatch')
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise RuntimeError('Checkpoint checksum mismatch')
    return path

def load_pretrained(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Wrong checkpoint version')
    ck=torch.load(path,map_location='cpu',weights_only=True);model=TabICL(ck['config']);model.load_state_dict(ck['state_dict'],strict=True)
    return model.eval()

def fit_numeric(x,threshold=4.):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Finite numeric context with at least two rows required')
    keep=np.array([len(np.unique(x[:,j]))>1 for j in range(x.shape[1])])
    if not keep.any():raise ValueError('No nonconstant context feature remains')
    x=x[:,keep];mean=x.mean(0);scale=x.std(0)+1e-6;z=np.clip((x-mean)/scale,-100,100)
    m=z.mean(0);s=np.maximum(z.std(0,ddof=1),1e-6);clean=np.where((z<m-threshold*s)|(z>m+threshold*s),np.nan,z)
    m=np.nanmean(clean,0);s=np.maximum(np.nanstd(clean,0,ddof=1),1e-6)
    return dict(keep=keep,mean=mean,scale=scale,lower=m-threshold*s,upper=m+threshold*s)

def transform_numeric(x,state):
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all() or x.ndim!=2 or x.shape[1]!=len(state['keep']):raise ValueError('Finite aligned numeric query required')
    z=np.clip((x[:,state['keep']]-state['mean'])/state['scale'],-100,100)
    z=np.maximum(-np.log1p(np.abs(z))+state['lower'],z)
    return np.minimum(np.log1p(np.abs(z))+state['upper'],z)

def predict_numeric(model,context,labels,query,temperature=.9):
    state=fit_numeric(context);classes,y=np.unique(labels,return_inverse=True)
    if not 2<=len(classes)<=10:raise ValueError('2–10 observed classes required')
    z=np.concatenate([transform_numeric(context,state),transform_numeric(query,state)])
    device=next(model.parameters()).device
    with torch.no_grad():logits=model(torch.tensor(z,device=device)[None],torch.tensor(y,device=device)[None])[0];p=(logits/temperature).softmax(-1)
    probabilities=p.cpu().numpy();probabilities/=probabilities.sum(1,keepdims=True)
    return probabilities,dict(preprocessed=z[None],classes=classes,kept_columns=np.flatnonzero(state['keep']),logits=logits.cpu().numpy())

def load_dataset(name):
    if name=='wdbc':
        from sklearn.datasets import load_breast_cancer
        data=load_breast_cancer();x,y=data.data,data.target
    else:
        from relkit.data import load_tier_a
        x,y=load_tier_a(name)
    _,y=np.unique(y,return_inverse=True)
    return np.asarray(x,dtype=np.float32),y

def model_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str((tuple(tensor.shape),tensor.dtype)).encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def stable_value(v):
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if v is Ellipsis:return 'Ellipsis'
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,(list,tuple)):return [stable_value(i) for i in v]
    if isinstance(v,dict):return {str(k):stable_value(i) for k,i in v.items()}
    if isinstance(v,(set,frozenset)):return sorted([stable_value(i) for i in v],key=repr)
    if isinstance(v,bytes):return v.hex()
    if isinstance(v,Path):return str(v)
    if isinstance(v,types.CodeType):return {k:stable_value(getattr(v,k)) for k in ['co_code','co_consts','co_names','co_varnames','co_freevars','co_cellvars','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_flags']}
    if inspect.isfunction(v):return dict(code=stable_value(v.__code__),defaults=stable_value(v.__defaults__),kwdefaults=stable_value(v.__kwdefaults__),closure=stable_value([c.cell_contents for c in v.__closure__ or []]))
    if inspect.isclass(v):return v.__module__+'.'+v.__qualname__
    raise TypeError('Unbound identity value '+str(type(v)))

def model_runtime_identity(model):
    """Bind actual module types, forward methods and non-weight runtime settings.

    Instance overrides and hooks are unsupported by this certified inference path;
    reject them rather than pretending an arbitrary callback has been audited.
    """
    records={}
    for name,module in model.named_modules():
        if 'forward' in vars(module):raise RuntimeError('Instance forward override invalidates source certification')
        if module._forward_hooks or module._forward_pre_hooks:raise RuntimeError('Forward hooks invalidate source certification')
        method=module.forward.__func__
        settings={k:stable_value(v) for k,v in vars(module).items() if not k.startswith('_')}
        records[name]=dict(type=type(module).__module__+'.'+type(module).__qualname__,forward=stable_value(method),settings=settings)
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def kernel_identity(namespace,root):
    """Bind callable roots, nested CodeTypes and their globals/default dependencies."""
    roots=['model_runtime_identity','attention_mix','context_keys','rotary_pairs','inducing_memory','conditional_affine','TabICL','load_pretrained','ensure_checkpoint','fit_numeric','transform_numeric','predict_numeric','load_dataset','run_experiment']
    seen={};allowed={namespace['TabICL'].__module__,namespace['run_experiment'].__module__}
    def visit(name,obj):
        if name in seen:return
        if inspect.isclass(obj):
            seen[name]={'class':obj.__qualname__}
            for k,v in vars(obj).items():
                if isinstance(v,(staticmethod,classmethod)):v=v.__func__
                if inspect.isfunction(v):visit(name+'.'+k,v)
        elif inspect.isfunction(obj):
            seen[name]=stable_value(obj)
            def dependency(v,n):
                if (inspect.isfunction(v) or inspect.isclass(v)) and (v.__module__ in allowed or v.__module__.startswith('relkit')):visit(n,v)
            def scan(code):
                for n in code.co_names:dependency(obj.__globals__.get(n),n)
                for c in code.co_consts:
                    if isinstance(c,types.CodeType):scan(c)
            scan(obj.__code__)
            for i,v in enumerate(obj.__defaults__ or []):dependency(v,name+'.default'+str(i))
            for k,v in (obj.__kwdefaults__ or {}).items():dependency(v,name+'.kwdefault.'+k)
    for name in roots:visit(name,namespace[name])
    from relkit.data import load_tier_a
    data_globals={k:stable_value(load_tier_a.__globals__[k]) for k in ['SPECS','CACHE']}
    result=dict(operators={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in seen.items()},constants={k:stable_value(namespace[k]) for k in ['MODEL_CONFIG','PROTOCOL','PRESETS66','CHECKPOINT_SHA','CHECKPOINT_URL','CHECKPOINT_FILE']},loader_globals=data_globals,loader_source_sha256=hashlib.sha256(Path(inspect.getfile(load_tier_a)).read_bytes()).hexdigest(),versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','scikit-learn']})
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest();return result

def run_experiment(root,model,config=None,namespace=None):
    ns=globals() if namespace is None else namespace;config=dict(PRESETS66['lab'] if config is None else config)
    identity=ns['kernel_identity'](ns,root);weights=ns['model_digest'](model);runtime=ns['model_runtime_identity'](model);records=[]
    for dataset in config['datasets']:
        x,y=ns['load_dataset'](dataset);ids=np.arange(len(y));data_sha=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        for seed in config['seeds']:
            pool,query=train_test_split(ids,test_size=PROTOCOL['query_fraction'],stratify=y,random_state=seed)
            # A fixed label-blind permutation gives genuinely nested context prefixes.
            pool=np.random.default_rng(seed+66000).permutation(pool)
            for fraction in PROTOCOL['context_fractions']:
                count=max(2,int(len(pool)*fraction));context=pool[:count]
                if len(np.unique(y[context]))!=len(np.unique(y[pool])):raise ValueError('A nested prefix missed a class; predeclare a different experiment')
                start=time.perf_counter();state=ns['fit_numeric'](x[context]);z=np.concatenate([ns['transform_numeric'](x[context],state),ns['transform_numeric'](x[query],state)])
                preprocessing_seconds=time.perf_counter()-start;device=next(model.parameters()).device
                tensor=torch.tensor(z,device=device)[None];labels=torch.tensor(y[context],device=device)[None]
                with torch.no_grad():
                    start=time.perf_counter();e=model.col_embedder(tensor,count);column_seconds=time.perf_counter()-start
                    start=time.perf_counter();r=model.row_interactor(e);row_seconds=time.perf_counter()-start
                    start=time.perf_counter();logits=model.icl_predictor(r,labels)[0,:,:len(np.unique(y[context]))];p=(logits/PROTOCOL['temperature']).softmax(-1).cpu().numpy();icl_seconds=time.perf_counter()-start
                record=dict(dataset=dataset,seed=seed,rows=len(y),features=x.shape[1],data_sha256=data_sha,fraction=fraction,context_ids=context.tolist(),query_ids=query.tolist(),context_labels=y[context].tolist(),targets=y[query].tolist(),probabilities=p.tolist(),logits=logits.cpu().tolist(),kept_columns=np.flatnonzero(state['keep']).tolist(),preprocessing_seconds=preprocessing_seconds,column_seconds=column_seconds,row_seconds=row_seconds,icl_seconds=icl_seconds,accuracy=float(accuracy_score(y[query],p.argmax(1))),log_loss=float(log_loss(y[query],p,labels=np.unique(y))))
                records.append(record);print(dataset,seed,count,round(record['log_loss'],5),flush=True)
    if ns['kernel_identity'](ns,root)['sha256']!=identity['sha256'] or ns['model_digest'](model)!=weights or ns['model_runtime_identity'](model)!=runtime:raise RuntimeError('Live code or weights changed during measurement')
    return dict(status='MEASURED_COMPLETE_PRETRAINED_TABICL',paper_reproduction='INCOMPARABLE',original_pretraining='NOT_RUN',paper_188_dataset_benchmark='NOT_RUN',large_500k='NOT_RUN',protocol=PROTOCOL,config=config,kernel_identity=identity,weights_sha256=weights,runtime_sha256=runtime,checkpoint_sha256=CHECKPOINT_SHA,records=records)

"""B08 visible course mechanism, inspired by LimiX-2 §2; not checkpoint compatible.
Scalar regression heads, fixed 4-column generator, 2 small blocks and MSE are
course choices. Released full architectures are separately archived/inlined.
"""
import hashlib, json, time
import numpy as np
import torch
from torch import nn

def sample_visibility(rows, support):
    """True means allowed: only support keys, for both support and queries."""
    if not 0 < support <= rows: raise ValueError('Need a nonempty support prefix')
    return (torch.arange(rows)[None, :] < support).expand(rows,rows).clone()

def feature_visibility(features, slots):
    """Feature queries see X and task slots; task queries see X only."""
    if min(features,slots)<1: raise ValueError('Need features and task slots')
    allowed=torch.ones(features+slots,features+slots,dtype=torch.bool)
    allowed[features:,features:]=False
    return allowed

def masked_mse(prediction, truth, mask):
    """Average squared error on scored cells; unscored values have zero gradient."""
    if prediction.shape!=truth.shape or mask.shape!=truth.shape or mask.dtype!=torch.bool:
        raise ValueError('Predictions, truth and Boolean mask must align')
    if not mask.any(): raise ValueError('Cannot score an empty mask')
    return ((prediction[mask]-truth[mask])**2).mean()

class Attention(nn.Module):
    """Single-head transparent attention: softmax(QK^T/sqrt(d))V."""
    def __init__(self,d):
        super().__init__();self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
    def forward(self,q,kv,allowed):
        scores=self.q(q)@self.k(kv).transpose(-1,-2)/(q.shape[-1]**.5)
        weights=scores.masked_fill(~allowed.to(q.device),float('-inf')).softmax(-1)
        return self.out(weights@self.v(kv))

class SwiGLU(nn.Module):
    def __init__(self,d):
        super().__init__();self.a=nn.Linear(d,2*d);self.b=nn.Linear(d,2*d);self.c=nn.Linear(2*d,d)
    def forward(self,x):return self.c(torch.nn.functional.silu(self.a(x))*self.b(x))

class DualAxis(nn.Module):
    """Separate sample-axis maps; concatenate K task slots across rows.
    Then independent feature/task SwiGLU and asymmetric within-row attention.
    """
    def __init__(self,d,k):
        super().__init__();self.d=d;self.k=k
        self.sx=Attention(d);self.sy=Attention(k*d);self.fx=Attention(d);self.fy=Attention(d)
        self.nx=nn.LayerNorm(d);self.ny=nn.LayerNorm(k*d);self.nf=nn.LayerNorm(d)
        self.gx=SwiGLU(d);self.gy=SwiGLU(k*d)
    def forward(self,x,t,support):
        b,n,f,d=x.shape;allowed=sample_visibility(n,support)
        xx=self.nx(x).transpose(1,2)
        x=x+self.sx(xx,xx,allowed).transpose(1,2)
        tt=t.reshape(b,n,self.k*d);norm=self.ny(tt)
        tt=tt+self.sy(norm,norm,allowed)
        x=x+self.gx(self.nx(x));tt=tt+self.gy(self.ny(tt));t=tt.reshape(b,n,self.k,d)
        tokens=self.nf(torch.cat([x,t],dim=2));access=feature_visibility(f,self.k)
        # Simultaneous updates: both use the same pre-feature-attention state.
        nx=x+self.fx(tokens[:,:,:f],tokens,access[:f])
        nt=t+self.fy(tokens[:,:,f:],tokens,access[f:])
        return nx,nt

class CourseModel(nn.Module):
    def __init__(self,d=16,k=4,features=4):
        super().__init__();self.d=d;self.k=k
        self.x_encoder=nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.Linear(d,d))
        self.y_encoder=nn.Linear(1,k*d)
        self.missing=nn.Parameter(torch.randn(d)*.02);self.query_mask=nn.Parameter(torch.randn(k,d)*.02)
        self.codes=nn.Parameter(torch.randn(features,4)*.02);self.code_map=nn.Linear(4,d,bias=False)
        self.blocks=nn.ModuleList([DualAxis(d,k) for _ in range(2)])
        self.x_head=nn.Linear(d,1);self.y_head=nn.Linear(k*d,1)
    def forward(self,x,y,hidden,support):
        if x.ndim!=3 or y.shape!=x.shape[:2] or hidden.shape!=x.shape:raise ValueError('Input shape mismatch')
        b,n,f=x.shape
        safe_x=x.masked_fill(hidden,0)
        xx=torch.where(hidden[...,None],self.missing,self.x_encoder(safe_x[...,None]))+self.code_map(self.codes)[None,None]
        observed=torch.arange(n,device=x.device)<support
        safe_y=torch.where(observed,y,torch.zeros_like(y))
        tt=self.y_encoder(safe_y[...,None]).reshape(b,n,self.k,self.d)
        tt=torch.where(observed[None,:,None,None],tt,self.query_mask)
        for depth,block in enumerate(self.blocks):
            xx,tt=block(xx,tt,support)
            if depth==0: reconstruction=self.x_head(xx).squeeze(-1)
        return self.y_head(tt.flatten(-2)).squeeze(-1),reconstruction

def make_episodes(seed,count,batch=4,rows=32,support=24):
    """Independent episodes: latent z -> correlated features -> noisy target.
    Distinct train/eval RNGs, same generator family. No distribution-shift claim.
    Mask exactly one of four feature cells per query; never mask support.
    """
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(count,batch,rows));e=rng.normal(size=(count,batch,rows,4))
    x=np.stack([z+.15*e[...,0],.8*z+.3*e[...,1],-.6*z+.4*e[...,2],e[...,3]],axis=-1).astype('float32')
    w=rng.normal(size=(count,batch,1,4)).astype('float32')/2
    y=((x*w).sum(-1)+.25*x[...,0]*x[...,1]+.1*rng.normal(size=z.shape)).astype('float32')
    hidden=np.zeros_like(x,dtype=bool)
    # One column per query chosen independently; 25% missingness in course only.
    columns=rng.integers(0,4,size=(count,batch,rows-support))
    for a in range(count):
        for b in range(batch):hidden[a,b,np.arange(support,rows),columns[a,b]]=True
    return dict(x=x,y=y,hidden=hidden)

def state_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train_arm(train,test,seed,objective,steps=120,support=24):
    """Complete fixed-step trainer; no validation selection or outcome-based retry."""
    if objective not in ['target','feature','combined']:raise ValueError('Unknown objective')
    torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=CourseModel();initial=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    curve=[];start=time.monotonic()
    for step in range(steps):
        x=torch.from_numpy(train['x'][step]);y=torch.from_numpy(train['y'][step]);hidden=torch.from_numpy(train['hidden'][step])
        p,r=model(x,y,hidden,support);target_mask=torch.zeros_like(y,dtype=torch.bool);target_mask[:,support:]=True
        ly=masked_mse(p,y,target_mask);lx=masked_mse(r,x,hidden)
        loss=ly if objective=='target' else lx if objective=='feature' else ly+lx
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%20==0 or step==steps-1:curve.append(dict(step=step+1,target=float(ly.detach()),feature=float(lx.detach())))
    model.eval();yp=[];xp=[]
    with torch.no_grad():
        for j in range(len(test['x'])):
            a,b=model(torch.from_numpy(test['x'][j]),torch.from_numpy(test['y'][j]),torch.from_numpy(test['hidden'][j]),support)
            yp.append(a.numpy());xp.append(b.numpy())
    return dict(y_pred=np.stack(yp),x_pred=np.stack(xp)),dict(seed=seed,objective=objective,initial_sha256=initial,final_sha256=state_hash(model),curve=curve,seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()))

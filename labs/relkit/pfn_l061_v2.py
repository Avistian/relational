"""L061 v2: from-scratch original row PFN on the paper's fixed RBF-GP prior.

Source: Müller et al. 2112.10510v7 §§3–5.1,D–F and original released
TransformersCanDoBayesianInference commit 9c20031b355923bdd456d5fcfe4e98092b016b97.
Original-style self-query mask, postnorm GELU blocks, zero residual projections,
full-support Riemann head. Budgets below are explicit new experiments, not paper runs.
"""
import base64,zlib,copy,hashlib,inspect,importlib.metadata,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

PRESETS={
 'smoke':dict(steps=30,batch=16,total=12,features=1,width=32,heads=4,hidden=64,layers=2,bins=32,lr=.001,seeds=[0],eval_tasks=32,contexts=[0,1,4,8,16]),
 'lab':dict(steps=2000,batch=32,total=17,features=1,width=64,heads=4,hidden=128,layers=3,bins=64,lr=.0003,seeds=[0,1,2],eval_tasks=256,contexts=[0,1,4,8,16,32]),
 'closer':dict(steps=10000,batch=32,total=65,features=5,width=128,heads=4,hidden=256,layers=6,bins=256,lr=.0003,seeds=[0,1,2],eval_tasks=1000,contexts=[0,1,4,16,32,64,128]),
}

def rbf_kernel(x,z,lengthscale=.6):
    """Unit output variance; last axes are rows and numeric features."""
    if lengthscale<=0:raise ValueError('lengthscale must be positive')
    return torch.exp(-((x.unsqueeze(-2)-z.unsqueeze(-3))**2).sum(-1)/(2*lengthscale**2))

def sample_gp(batch,total,features,generator,lengthscale=.6,noise=1e-4,device='cpu'):
    """Sample joint noisy GP labels. Double Cholesky prevents tiny-noise instability."""
    if min(batch,total,features)<1 or noise<=0:raise ValueError('Positive dimensions and observation-noise variance required')
    # Draw on CPU for a device-independent task stream; transfer after stable sampling.
    x=torch.rand(batch,total,features,generator=generator,dtype=torch.float64)
    covariance=rbf_kernel(x,x,lengthscale)+noise*torch.eye(total,dtype=x.dtype)
    z=torch.randn(batch,total,1,generator=generator,dtype=x.dtype)
    y=(torch.linalg.cholesky(covariance)@z).squeeze(-1)
    return x.float().to(device),y.float().to(device)

def gp_posterior(x,context_y,n,lengthscale=.6,noise=1e-4):
    """Exact marginal PPD for observed query y; all computations in float64."""
    if not 0<=n<x.shape[1] or context_y.shape!=x.shape[:1]+(n,):raise ValueError('Need B×n context labels and at least one query')
    x=x.double();context_y=context_y.double();xc,xq=x[:,:n],x[:,n:]
    if n==0:return torch.zeros(xq.shape[:2],device=x.device,dtype=x.dtype),torch.full(xq.shape[:2],1+noise,device=x.device,dtype=x.dtype)
    kcc=rbf_kernel(xc,xc,lengthscale)+noise*torch.eye(n,device=x.device,dtype=x.dtype)
    kcq=rbf_kernel(xc,xq,lengthscale);chol=torch.linalg.cholesky(kcc)
    alpha=torch.cholesky_solve(context_y.unsqueeze(-1),chol)
    mean=(kcq.transpose(-2,-1)@alpha).squeeze(-1)
    solved=torch.linalg.solve_triangular(chol,kcq,upper=False)
    variance=1+noise-solved.square().sum(-2)
    if torch.any(variance<=0):raise ArithmeticError('Nonpositive GP variance: check solve and noise convention')
    return mean,variance

def context_attention_mask(total,n,device='cpu'):
    """Additive mask: all context columns plus identity, as original source."""
    if not 0<=n<total:raise ValueError('Need 0≤n<total')
    allowed=torch.zeros(total,total,dtype=torch.bool,device=device)
    allowed[:,:n]=True
    allowed|=torch.eye(total,dtype=torch.bool,device=device)
    return torch.zeros(total,total,device=device).masked_fill(~allowed,float('-inf'))

def attention_mix(q,k,v,mask):
    """B×H×N×Dh projections -> same-shaped values mixed over sender rows."""
    scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
    weights=torch.softmax(scores+mask,dim=-1)
    return weights@v

class PFNBlock(nn.Module):
    """Visible postnorm TransformerEncoderLayer; dropout=0 for this mirror."""
    def __init__(self,width,heads,hidden):
        super().__init__()
        if width%heads:raise ValueError('Heads must divide width')
        self.heads=heads;self.width=width
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.ff1=nn.Linear(width,hidden);self.ff2=nn.Linear(hidden,width)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        nn.init.xavier_uniform_(self.qkv.weight);nn.init.zeros_(self.qkv.bias)
        for layer in [self.out,self.ff2]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,h,mask):
        b,n,d=h.shape
        q,k,v=self.qkv(h).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        mixed=attention_mix(q,k,v,mask).transpose(1,2).reshape(b,n,d)
        h=self.norm1(h+self.out(mixed))
        return self.norm2(h+self.ff2(torch.nn.functional.gelu(self.ff1(h))))

class RowPFN(nn.Module):
    """Whole rows are tokens; query labels never enter the forward signature."""
    def __init__(self,features=1,width=64,heads=4,hidden=128,layers=3,bins=64):
        super().__init__()
        self.x_encoder=nn.Linear(features,width);self.y_encoder=nn.Linear(1,width)
        block=PFNBlock(width,heads,hidden)
        # Original nn.TransformerEncoder clones one initialized template.
        self.blocks=nn.ModuleList([copy.deepcopy(block) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,bins))
    def forward(self,x,context_y):
        n=context_y.shape[1]
        if x.ndim!=3 or context_y.shape[0]!=x.shape[0] or n>=x.shape[1]:raise ValueError('Expected B×N×F x and B×n labels, n<N')
        h=self.x_encoder(x)
        h=torch.cat([h[:,:n]+self.y_encoder(context_y.unsqueeze(-1)),h[:,n:]],dim=1)
        mask=context_attention_mask(x.shape[1],n,x.device)
        for block in self.blocks:h=block(h,mask)
        return self.head(h[:,n:])

def make_borders(bins,seed=731,samples=65536):
    """Prior marginal quantiles, never fitted on evaluation labels. See tail caveat."""
    generator=torch.Generator().manual_seed(seed)
    ys=torch.randn(samples,generator=generator)*math.sqrt(1.0001)
    return torch.quantile(ys,torch.linspace(0,1,bins+1))

def riemann_nll(logits,y,borders):
    """Full-support density: uniform interior bars, weighted half-normal tails."""
    if logits.shape[:-1]!=y.shape or logits.shape[-1]!=borders.numel()-1:raise ValueError('Logit/target/border shapes disagree')
    widths=borders[1:]-borders[:-1]
    if len(widths)<2 or torch.any(widths<=0):raise ValueError('Need at least two positive-width bars')
    index=(torch.searchsorted(borders,y.contiguous())-1).clamp(0,len(widths)-1)
    log_mass=torch.log_softmax(logits,-1).gather(-1,index.unsqueeze(-1)).squeeze(-1)
    log_density=log_mass-torch.log(widths[index])
    # HalfNormal(1).median = sqrt(2)*erfinv(.5); width contains half a tail's mass.
    unit_median=math.sqrt(2)*torch.erfinv(borders.new_tensor(.5))
    for bucket,anchor,sign in [(0,borders[1],-1),(-1,borders[-2],1)]:
        scale=widths[bucket]/unit_median
        distance=(sign*(y-anchor)).clamp(min=0)
        component=.5*math.log(2/math.pi)-torch.log(scale)-.5*(distance/scale)**2
        log_density=torch.where(index==(bucket%len(widths)),log_mass+component,log_density)
    return -log_density

def riemann_mean(logits,borders):
    widths=borders[1:]-borders[:-1];means=(borders[1:]+borders[:-1])/2
    median=math.sqrt(2)*torch.erfinv(borders.new_tensor(.5))
    means=means.clone();means[0]=borders[1]-widths[0]/median*math.sqrt(2/math.pi)
    means[-1]=borders[-2]+widths[-1]/median*math.sqrt(2/math.pi)
    return logits.softmax(-1)@means

def riemann_cdf(logits,y,borders):
    """Integrate each normalized component; useful for central coverage and plots."""
    widths=borders[1:]-borders[:-1]
    fractions=((y.unsqueeze(-1)-borders[:-1])/widths).clamp(0,1)
    median=math.sqrt(2)*torch.erfinv(borders.new_tensor(.5))
    fractions[...,0]=1-torch.erf(((borders[1]-y).clamp(min=0))/(widths[0]/median*math.sqrt(2)))
    fractions[...,-1]=torch.erf(((y-borders[-2]).clamp(min=0))/(widths[-1]/median*math.sqrt(2)))
    return (logits.softmax(-1)*fractions).sum(-1)

def train_pfn(config,seed,device='cpu'):
    """Algorithm 1: independent fresh datasets; sampled y targets only, never PPD."""
    torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+10000)
    model=RowPFN(**{k:config[k] for k in ['features','width','heads','hidden','layers','bins']}).to(device)
    borders=make_borders(config['bins']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    total=config['total'];weights=1/torch.arange(total,0,-1,dtype=torch.float32)
    trace=[];start=time.perf_counter();model.train()
    for step in range(config['steps']):
        n=int(torch.multinomial(weights,1,generator=generator))
        x,y=sample_gp(config['batch'],total,config['features'],generator,device=device)
        warmup=max(1,config['steps']//10)
        factor=min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,config['steps']-warmup)))
        for group in optimizer.param_groups:group['lr']=config['lr']*factor
        logits=model(x,y[:,:n]);loss=riemann_nll(logits,y[:,n:],borders).mean()
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training likelihood')
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==0 or (step+1)%max(1,config['steps']//10)==0:
            trace.append(dict(step=step+1,n_context=n,nll=loss.item(),lr=group['lr']))
    return model.eval(),borders,trace,time.perf_counter()-start

def encode_tensor(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),zlib_base64=base64.b64encode(zlib.compress(array.tobytes())).decode(),sha256=hashlib.sha256(array.tobytes()).hexdigest())

def evaluate_pfn(model,borders,config,seed):
    """Fixed independent evaluation task stream, paired across training seeds."""
    records=[];device=next(model.parameters()).device
    with torch.no_grad():
        for lengthscale,regime in [(.6,'matched'),(.1,'prior_shift')]:
            for n in config['contexts']:
                generator=torch.Generator().manual_seed(20000+n+(1000 if regime=='prior_shift' else 0))
                audit=[]
                values={k:[] for k in ['pfn_nll','gp_nll','prior_nll','mean_squared_error','coverage95','excess_nll']}
                for start in range(0,config['eval_tasks'],64):
                    batch=min(64,config['eval_tasks']-start)
                    x,y=sample_gp(batch,n+1,config['features'],generator,lengthscale=lengthscale,device=device)
                    logits=model(x,y[:,:n]);mu,var=gp_posterior(x,y[:,:n],n,lengthscale)
                    target=y[:,n:].double();pfn=riemann_nll(logits,y[:,n:],borders).double()
                    gp=.5*(torch.log(2*math.pi*var)+(target-mu)**2/var)
                    prior=.5*(math.log(2*math.pi*1.0001)+target**2/1.0001)
                    cdf=riemann_cdf(logits,y[:,n:],borders)
                    outputs=dict(pfn_nll=pfn,gp_nll=gp,prior_nll=prior,mean_squared_error=(riemann_mean(logits,borders)-mu)**2,coverage95=((cdf>=.025)&(cdf<=.975)).double(),excess_nll=pfn-gp)
                    for key,value in outputs.items():values[key]+=value.flatten().cpu().tolist()
                    audit.append({key:encode_tensor(value) for key,value in dict(x=x,y=y,logits=logits,gp_mean=mu,gp_variance=var).items()})
                records.append(dict(seed=seed,regime=regime,n_context=n,task_seed=20000+n+(1000 if regime=='prior_shift' else 0),audit_batches=audit,task_values=values,**{k:float(np.mean(v)) for k,v in values.items()}))
    return records

def summarize_records(records):
    """Seed SD and paired-task intervals answer different questions; retain both."""
    from scipy.stats import t
    summaries=[]
    for regime,n in sorted({(r['regime'],r['n_context']) for r in records}):
        rows=[r for r in records if (r['regime'],r['n_context'])==(regime,n)]
        detail=dict(regime=regime,n_context=n,seeds=[r['seed'] for r in rows])
        for metric in ['pfn_nll','gp_nll','prior_nll','mean_squared_error','coverage95','excess_nll']:
            seed_values=np.array([r[metric] for r in rows]);tasks=np.array([r['task_values'][metric] for r in rows]).mean(0)
            detail[metric]=dict(mean=float(seed_values.mean()),seed_sd=float(seed_values.std(ddof=1)) if len(rows)>1 else None,conditional_task_t95=float(t.ppf(.975,len(tasks)-1)*tasks.std(ddof=1)/math.sqrt(len(tasks))),seed_values=seed_values.tolist())
        summaries.append(detail)
    return summaries

def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run_experiment(preset='lab',output=None,device='cpu'):
    """No resume or result overwrite. Notebook replaces repository identity explicitly."""
    if preset=='paper':raise ValueError('Full original Fig4 protocol is not implemented; closer is a documented scale step')
    if preset not in PRESETS:raise ValueError('Choose smoke, lab or closer')
    output=Path(output) if output is not None else None
    if output is not None and output.exists():raise FileExistsError('Choose a fresh evidence path')
    config=copy.deepcopy(PRESETS[preset]);records=[];runs=[];start=time.perf_counter();torch.set_num_threads(1)
    for seed in config['seeds']:
        model,borders,trace,seconds=train_pfn(config,seed,device)
        records+=evaluate_pfn(model,borders,config,seed)
        runs.append(dict(seed=seed,training_seconds=seconds,trace=trace,weights_sha256=state_hash(model)))
        print(f'L061 {preset} seed {seed}: {seconds:.1f}s training',flush=True)
    # Persist a transparent two-observation prediction trace for the portable figure.
    with torch.no_grad():
        grid=torch.linspace(0,1,101,device=device).view(1,-1,1).repeat(1,1,config['features'])
        context=torch.tensor([.2,.8],device=device).view(1,2,1).repeat(1,1,config['features'])
        x=torch.cat([context,grid],1);cy=torch.tensor([[1.,-1.]],device=device)
        logits=model(x,cy);mu,var=gp_posterior(x,cy,2)
        curve=dict(training_seed=config['seeds'][-1],context_x=[.2,.8],context_y=[1.,-1.],grid=grid[0,:,0].cpu().tolist(),logits=logits[0].cpu().tolist(),gp_mean=mu[0].cpu().tolist(),gp_variance=var[0].cpu().tolist(),pfn_mean=riemann_mean(logits,borders)[0].cpu().tolist())
    result=dict(status='COMPLETE',operator='pfn_l061_v2',preset=preset,config=config,prior=dict(kernel='RBF',lengthscale=.6,noise_variance=1e-4,output_variance=1.,x='Uniform unit cube'),runs=runs,records=records,summary=summarize_records(records),borders=borders.cpu().tolist(),curve=curve,seconds=time.perf_counter()-start,versions={k:importlib.metadata.version(k) for k in ['numpy','torch','scipy']},paper_reproduction='INCOMPARABLE',repository_reference_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_hashes_scope='Repository reference file; notebook actual kernel source is captured separately',uncertainty='Training-seed SD over fixed tasks; paired task t interval conditional on these trained seeds; tasks shared across training seeds',deviations=[f"{config['features']} input features vs 5 in Fig4",f"width{config['width']}/layers{config['layers']}/bins{config['bins']} vs released width512/layers6/bins1000 or10000",f"{config['steps']} updates per seed; N{config['total']} not N2010",f"{config['eval_tasks']} evaluation tasks vs1000",'sampled marginal quantile borders; 10percent warmup per step','No MCMC/VI/BNN/tabular/Omniglot replication'])
    if output is not None:output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

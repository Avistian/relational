"""Portable executable audit of pinned release; no network or training."""
from pathlib import Path
import ast,hashlib,json,math,random,typing,importlib.util
from collections import defaultdict,deque
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import duckdb

def audit184(packet,manifest,gaussian):
    packet=Path(packet)
    for name,digest in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed authenticated input: '+name)
    torch.set_num_threads(1)
    src=packet/'upstream'
    # Compile the original sampling functions unchanged. Only the process pool
    # is replaced by an in-process executor, to keep this diagnostic portable.
    tree=ast.parse((src/'utils.py').read_text())
    wanted={'gather_1_and_2_hop_with_seed_time','_process_one_seed','local_nodes_hetero'}
    code=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[])
    class SerialPool:
        def __init__(self,processes,initializer,initargs):initializer(*initargs)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,args):return list(map(fn,args))
    ns=dict(vars(typing));ns.update(np=np,torch=torch,random=random,defaultdict=defaultdict,deque=deque,HeteroData=object,Pool=SerialPool)
    adj={'driver':[{('race',0),('race',1)}], 'race':[{('driver',0)},{('driver',0)}]}
    def init(adj,nodes):ns.update(GLOBAL_ADJ=adj,GLOBAL_ALL_NODES=nodes)
    ns['init_worker_globals']=init;init(adj,[])
    exec(compile(code,'pinned-utils.py','exec'),ns)
    day=86400.;data={'driver':SimpleNamespace(),'race':SimpleNamespace(time=torch.tensor([5*day,15*day]))}
    args=(data,3,'driver',0)
    early=ns['_process_one_seed']((*args,10*day,42))[2]
    late=ns['_process_one_seed']((*args,20*day,42))[2]
    collapsed=ns['local_nodes_hetero'](data,3,('driver',torch.tensor([0,0])),torch.tensor([10*day,20*day]),num_workers=1)
    shared=collapsed['driver'][0][0]
    assert len(early)==2 and len(late)==3 and len(shared)==3
    assert len(collapsed['driver'])==1
    leak=[n[1] for n in shared if n[0]=='race' and data['race'].time[n[1]].item()>10*day]
    assert leak==[1]
    # Execute actual source attention in float64 and compare independent kernels,
    # then reconstruct a full EncoderLayer with explicit matrix multiplication.
    mod=SimpleNamespace(__name__='l184_pinned_local')
    exec(compile((src/'local_module.py').read_text(),'pinned-local_module.py','exec'),mod.__dict__)
    torch.manual_seed(184)
    lm=mod.LocalModule(3,8,n_layers=1,num_heads=2,hidden_dim=8,dropout_rate=0,attention_dropout_rate=0,max_time_days=4).double().eval()
    x=torch.randn(2,3,8,dtype=torch.float64,requires_grad=True);times=torch.tensor([[0.,2.,4.],[0.,1.,3.]],dtype=torch.float64)
    captured={}
    handle=lm.layers[0].register_forward_pre_hook(lambda m,a,kw:captured.update(bias=kw['attn_bias']),with_kwargs=True)
    y=lm(x,neighbor_times=times,pretrain_token=True);handle.remove()
    diff=abs(times.numpy()[:,:,None]-times.numpy()[:,None,:])
    kernels=np.stack([gaussian(diff,float(mu),abs(float(sig))+1e-5) for mu,sig in zip(lm.kernel_mu.detach(),lm.kernel_sigma.detach())],axis=-1)
    expected=kernels@lm.kernel_proj.weight.detach().numpy().T+lm.kernel_proj.bias.detach().numpy()
    kernel_error=float(np.max(abs(expected.transpose(0,3,1,2)-captured['bias'].detach().numpy())))
    assert kernel_error<1e-12
    layer=lm.layers[0];z=lm.att_embeddings_nope(x);norm=layer.self_attention_norm(z)
    q,k,v=[f(norm).view(2,3,2,4).transpose(1,2) for f in (layer.q_proj,layer.k_proj,layer.v_proj)]
    w=torch.softmax(q@k.transpose(-1,-2)/2+captured['bias'],dim=-1)
    a=(w@v).transpose(1,2).reshape(2,3,8)
    z=z+layer.out_proj(a);manual=lm.final_ln(z+layer.ffn(layer.ffn_norm(z)))
    forward_error=float((manual-y).abs().max().detach());assert forward_error<1e-12
    g1=torch.autograd.grad(y.square().sum(),x,retain_graph=True)[0]
    g2=torch.autograd.grad(manual.square().sum(),x)[0]
    grad_error=float((g1-g2).abs().max());assert grad_error<1e-10
    # All released driver-position labels: raw join independent of task code.
    results=pd.read_parquet(packet/'db/results.parquet');drivers=pd.read_parquet(packet/'db/drivers.parquet')
    counts={};collision={};label_error=0.;total=0
    for split in ['train','val','test']:
        truth=pd.read_parquet(packet/'task'/f'{split}.parquet')
        assert not truth.duplicated(['driverId','date']).any()
        timestamps=pd.DataFrame({'cutoff':truth.date.unique()})
        rebuilt=duckdb.sql('''SELECT t.cutoff AS date,d.driverId,avg(r.positionOrder) AS position
         FROM timestamps t JOIN results r ON r.date>t.cutoff AND r.date<=t.cutoff+INTERVAL '60 days'
         JOIN drivers d ON d.driverId=r.driverId
         WHERE d.driverId IN (SELECT driverId FROM results WHERE date>t.cutoff-INTERVAL '1 year')
         GROUP BY t.cutoff,d.driverId''').df()
        joined=truth.merge(rebuilt,on=['driverId','date'],how='outer',suffixes=('_y','_sql'),indicator=True)
        assert (joined['_merge']=='both').all()
        error=float(abs(joined.position_y-joined.position_sql).max());assert error<1e-12
        label_error=max(label_error,error);counts[split]=len(truth);total+=len(truth)
        # Source chunk size is 10000; all three complete populations fit one chunk.
        assert len(truth)<=10000
        collision[split]={'queries':len(truth),'unique_entities':int(truth.driverId.nunique()),
           'overwritten_query_entries':int(len(truth)-truth.driverId.nunique()),
           'rows_sharing_entity':int(truth.duplicated('driverId',keep=False).sum())}
    return {'audit':'PASS','selected_experiment':'INCOMPLETE_SOURCE_TEMPORAL_GATE','fresh_training':'NOT_RUN',
      'paper_target_mae':3.7345,'measured_model_mae':None,'whole_paper':'NOT_RUN','historical_identity':'NOT_ESTABLISHED',
      'labels':total,'counts':counts,'label_max_error':label_error,'cache_collisions':collision,
      'synthetic_source_probe':{'input_queries':2,'output_cache_entries':1,'early_legal_tokens':len(early),'reused_late_tokens':len(shared),'future_nodes_in_early_context':leak},
      'kernel_max_error':kernel_error,'attention_forward_max_error':forward_error,'attention_input_gradient_max_error':grad_error,
      'scope':'Original sampling functions and attention module executed; full model/trainer NOT_RUN. Synthetic leak is a counterexample, not a measured full-data leak rate.',
      'cloud_usd':0,'learner':'PENDING_WRITTEN_DEFENSE'}

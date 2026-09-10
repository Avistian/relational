"""Closer numeric TabM-mini procedure. Resource presets are NOT fidelity certificates."""
import argparse, copy, hashlib, inspect, json, time, importlib.metadata, platform, sys, types
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from _live_identity_l051 import code_fingerprint
from relkit.realmlp_experiment import load_task, error
from relkit.tabr_experiment import seed_interval
from relkit.tabm_v2 import TabM, member_mean_loss, ensemble_predict
PRESETS={
 'smoke':dict(train_cap=256,eval_cap=128,width=32,k=4,epochs=3,seeds=[0],batch_size=64),
 'closer':dict(train_cap=6000,eval_cap=None,width=256,k=32,epochs=150,seeds=[0,1,2],batch_size=256),
 'paper':dict(train_cap=None,eval_cap=None,width=512,k=32,epochs=1000,seeds=list(range(15)),batch_size=256),
}
def fit_closer(data, seed, width, k, epochs, batch_size, device='cpu', model_class=None,
               loss_fn=None, predict_fn=None):
    """App D.2 and §3.4: each member receives a full independently shuffled batch."""
    model_class=TabM if model_class is None else model_class
    loss_fn=member_mean_loss if loss_fn is None else loss_fn
    predict_fn=ensemble_predict if predict_fn is None else predict_fn
    torch.manual_seed(seed);reg=data['regression']
    # Approximate paper default; released modified quantile/noise is not matched.
    transform=QuantileTransformer(n_quantiles=min(1000,len(data['raw']['train'])),
        output_distribution='normal',subsample=None,random_state=seed).fit(data['raw']['train'])
    x={s:torch.tensor(transform.transform(v),dtype=torch.float32,device=device) for s,v in data['raw'].items()}
    y={s:torch.tensor(v,dtype=torch.float32 if reg else torch.long,device=device) for s,v in data['y'].items()}
    model=model_class(x['train'].shape[1],k=k,width=width,depth=3,dropout=.1,regression=reg,arch='mini',seed=seed).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
    best=float('inf');state=None;best_epoch=0;history=[];start=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        permutations=torch.stack([torch.randperm(len(x['train']),device=device) for _ in range(k)],dim=1)
        for idx in permutations.split(batch_size):
            out=model(x['train'][idx])              # [B,k,d_y], targets [B,k]
            # Flatten BOTH in row-major order, preserving each output/target pair.
            loss=loss_fn(out.reshape(-1,1,out.shape[-1]),y['train'][idx].reshape(-1),reg)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite member loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():
            p=torch.cat([predict_fn(model(b),reg) for b in x['val'].split(256)]).cpu().numpy()
        score=error(p,data['y']['val'],reg,data['target_std']);history.append(score)
        if score<best:best=score;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict())
        if epoch+1-best_epoch>=16:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():p=torch.cat([predict_fn(model(b),reg) for b in x['test'].split(256)]).cpu().numpy()
    return dict(seed=seed,error=error(p,data['y']['test'],reg,data['target_std']),prediction=p.tolist(),
        best_epoch=best_epoch,history=history,epochs_run=len(history),seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in model.parameters()))
def live_identity(*roots):
    def names(code):
        found=set(code.co_names)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):found.update(names(value))
        return found
    pending=list(roots);seen=set();found={}
    while pending:
        obj=pending.pop()
        if id(obj) in seen:continue
        seen.add(id(obj))
        if inspect.isclass(obj):pending.extend(v for v in vars(obj).values() if inspect.isfunction(v))
        elif inspect.isfunction(obj):
            found[obj.__qualname__]=code_fingerprint(obj)
            for name in names(obj.__code__):
                value=obj.__globals__.get(name);module=getattr(value,'__module__','') or ''
                if (inspect.isclass(value) or inspect.isfunction(value)) and (module=='__main__' or module.startswith(('relkit.tabm','relkit.realmlp','_paper_repro_l054'))):pending.append(value)
    return dict(sorted(found.items()))
def reproduce(preset='closer',out='artifacts/l054/closer',device='cpu',model_class=TabM,
              loss_fn=member_mean_loss,predict_fn=ensemble_predict,runner=fit_closer):
    torch.set_num_threads(1);config=PRESETS[preset];data=load_task('california',config['train_cap'],config['eval_cap'],Path(__file__).resolve().parent/'data/cache/l052')
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    payload=dict(config=config,data_hashes=data['hashes'],selection=data['selection'],
        functions=live_identity(model_class,loss_fn,predict_fn,runner,load_task,error),device=device,
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn']},
        python=sys.version,platform=platform.machine())
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    identity=folder/'identity.json'
    if identity.exists() and json.loads(identity.read_text())['sha256']!=fingerprint:raise RuntimeError('Resume identity changed; choose a new output directory')
    identity.write_text(json.dumps(dict(sha256=fingerprint,**payload),indent=1));runs=[]
    for seed in config['seeds']:
        path=folder/f'seed-{seed}.json'
        if path.exists():run=json.loads(path.read_text())
        else:
            run=runner(data,seed,**{a:config[a] for a in ['width','k','epochs','batch_size']},device=device,
                           model_class=model_class,loss_fn=loss_fn,predict_fn=predict_fn)
            path.write_text(json.dumps(run))
        runs.append(run)
    result=dict(preset=preset,config=config,metric='RMSE',summary=seed_interval([r['error'] for r in runs]),
        runs=runs,identity=fingerprint,verdict='INCOMPARABLE',
        ledger=dict(verified_here='numeric mini parity and separate local v2 results',
                    paper_claim='46 datasets, TPE selection, mostly15 seeds; NOT_REPRODUCED',
                    scale_up='measured fixed-recipe California numeric procedure; INCOMPARABLE'),
        deviations=['One dataset','Capped train in smoke/closer','Fixed hyperparameters, no TPE',
                    'Unmodified sklearn normal quantiles, no author jitter','No feature embeddings',
                    'Epoch ceiling; RNG/library versions differ','Numeric mini, not full/dagger variants'])
    (folder/'result.json').write_text(json.dumps(result,indent=1));return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preset',choices=PRESETS,default='smoke');p.add_argument('--out');p.add_argument('--device',default='cpu');a=p.parse_args()
    result=reproduce(a.preset,a.out or str(Path(__file__).resolve().parent/'artifacts/l054'/a.preset),a.device);print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

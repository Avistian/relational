"""Year 2 exit comparison. Full local experiment, not a published benchmark.

All four families receive identical numeric/binary columns and row partitions.
The complete historical v2 weights are frozen. Only its temperature is selected.
"""
import copy
import hashlib
import importlib.metadata
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from relkit.ft_l080 import Transformer
from relkit.tabm_v2 import TabM
from relkit.tabpfn_l064_v2 import ensure_checkpoint, load_pretrained, predict_numeric, CHECKPOINT_SHA

ARMS=['XGBoost','FT-Transformer','TabM','TabPFN-v2']
TASKS=['ecom-offers','homesite-insurance']
PRESETS={'smoke':dict(train_cap=64,eval_cap=24,epochs=2,trees=20,seeds=[0]),
         'exam':dict(train_cap=128,eval_cap=48,epochs=16,trees=120,seeds=[0,1,2]),
         'extended':dict(train_cap=512,eval_cap=192,epochs=128,trees=800,seeds=[0,1,2])}

def choose_validation(errors):
    a=np.asarray(errors,float)
    if a.ndim!=1 or a.size==0 or not np.isfinite(a).all(): raise ValueError('Finite nonempty validation vector required')
    return int(np.argmin(a))

def binary_loss(y,p):
    y,p=np.asarray(y),np.asarray(p,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Aligned binary labels and finite probabilities in [0,1] required')
    p=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))

def load_panel(root,preset):
    """Frozen portable, hash-pinned raw subset; preprocessing is fitted later."""
    root=Path(root);folder=root/'data/l080';meta=json.loads((folder/f'{preset}.json').read_text())
    path=folder/f'{preset}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['npz_sha256']: raise ValueError('Raw data checksum changed')
    arrays=np.load(path,allow_pickle=False)
    return arrays,meta

def fit_candidate(arm,train,ytrain,val,yval,seed,candidate,cfg):
    """Training and early stopping cannot access test features or labels."""
    torch.manual_seed(seed)
    if arm=='XGBoost':
        m=XGBClassifier(n_estimators=cfg['trees'],max_depth=[3,6][candidate],learning_rate=.05,
            n_jobs=1,random_state=seed,subsample=.8,colsample_bytree=.8,tree_method='hist')
        m.fit(train,ytrain)
        predict=lambda z:m.predict_proba(z)[:,list(m.classes_).index(1)].astype(float)
        vp=predict(val)
        return predict,vp,dict(best_epoch=None,history=[binary_loss(yval,vp)])
    scaler=StandardScaler().fit(train)
    tr=torch.tensor(scaler.transform(train),dtype=torch.float32)
    va=torch.tensor(scaler.transform(val),dtype=torch.float32);yt=torch.tensor(ytrain,dtype=torch.long)
    if arm=='FT-Transformer':
        m=Transformer(d_numerical=tr.shape[1],categories=None,token_bias=True,n_layers=3,d_token=32,n_heads=4,
            d_ffn_factor=4/3,attention_dropout=.1,ffn_dropout=.1,residual_dropout=0.,activation='reglu',
            prenormalization=True,initialization='kaiming',kv_compression=None,kv_compression_sharing=None,d_out=1)
    elif arm=='TabM': m=TabM(tr.shape[1],width=64,depth=3,k=8,arch='tabm',seed=seed,dropout=.1)
    else: raise ValueError(arm)
    opt=torch.optim.AdamW(m.parameters(),lr=[.001,.003][candidate],weight_decay=1e-4)
    def tensor_predict(z):
        m.eval()
        with torch.no_grad():
            return torch.cat([m.predict(b)[:,1] if arm=='TabM' else m(b,None).sigmoid() for b in z.split(32)]).numpy().astype(float)
    best=float('inf');state=None;history=[];best_epoch=None
    for epoch in range(cfg['epochs']):
        m.train()
        for ids in torch.randperm(len(tr)).split(32):
            loss=m.member_losses(tr[ids],yt[ids]) if arm=='TabM' else nn.functional.binary_cross_entropy_with_logits(m(tr[ids],None),yt[ids].float())
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            opt.zero_grad();loss.backward();opt.step()
        error=binary_loss(yval,tensor_predict(va));history.append(error)
        if error<best: best=error;state=copy.deepcopy(m.state_dict());best_epoch=epoch+1
    m.load_state_dict(state)
    predict=lambda z:tensor_predict(torch.tensor(scaler.transform(z),dtype=torch.float32))
    return predict,predict(val),dict(best_epoch=best_epoch,history=history,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())

def temperature_probability(logits,temperature):
    z=np.asarray(logits,float)/temperature;z-=z.max(1,keepdims=True);p=np.exp(z);return (p/p.sum(1,keepdims=True))[:,1]

def run_experiment(root,preset,output,namespace=None):
    """No resume/overwrite. A failed run remains visibly IN_PROGRESS."""
    s=globals() if namespace is None else namespace
    root=Path(root);out=Path(output)
    if out.exists(): raise FileExistsError('Choose a new evidence output path')
    out.parent.mkdir(parents=True,exist_ok=True);cfg=copy.deepcopy(PRESETS[preset])
    torch.set_num_threads(1);arrays,meta=s['load_panel'](root,preset)
    r=dict(status='IN_PROGRESS',lesson=80,preset=preset,config=cfg,arms=ARMS,tasks=TASKS,records=[],data=meta,
        scope='Bounded numeric/binary local comparison; original paper benchmarks INCOMPARABLE / NOT_RUN',
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','xgboost']},
        checkpoint_sha256=CHECKPOINT_SHA,source_hashes={f:hashlib.sha256((root/'relkit'/f).read_bytes()).hexdigest() for f in ['exit_l080.py','ft_l080.py','tabm_v2.py','tabpfn_l064_v2.py']})
    out.write_text(json.dumps(r,indent=2)+'\n')
    model,_=s['load_pretrained'](s['ensure_checkpoint'](root));start=time.perf_counter()
    with threadpool_limits(limits=1):
        for name,regime in itertools.product(TASKS,['random','temporal']):
            key=name+'/'+regime
            raw={part:arrays[key+'/'+part+'/x'] for part in ['train','val','test']}
            y={part:arrays[key+'/'+part+'/y'] for part in raw}
            imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(raw['train'])
            x={part:imputer.transform(v).astype('float32') for part,v in raw.items()}
            for seed,arm in itertools.product(cfg['seeds'],ARMS):
                t=time.perf_counter();candidate_records=[]
                if arm=='TabPFN-v2':
                    # Fix one complete query batch per partition. No test labels in context.
                    _,audit=s['predict_numeric'](model,x['train'],y['train'],x['val'],seed=seed)
                    for temp in [.7,1.]:
                        vp=s['temperature_probability'](audit['logits'],temp)
                        candidate_records.append(dict(validation_predictions=vp.tolist(),validation_loss=s['binary_loss'](y['val'],vp),temperature=temp,best_epoch=None))
                    chosen=s['choose_validation']([c['validation_loss'] for c in candidate_records])
                    _,test_audit=s['predict_numeric'](model,x['train'],y['train'],x['test'],seed=seed)
                    pred=s['temperature_probability'](test_audit['logits'],candidate_records[chosen]['temperature'])
                else:
                    predictors=[]
                    for candidate in range(2):
                        fn,vp,details=s['fit_candidate'](arm,x['train'],y['train'],x['val'],y['val'],seed,candidate,cfg)
                        predictors.append(fn);candidate_records.append(dict(validation_predictions=vp.tolist(),validation_loss=s['binary_loss'](y['val'],vp),**details))
                    chosen=s['choose_validation']([c['validation_loss'] for c in candidate_records]);pred=predictors[chosen](x['test'])
                r['records'].append(dict(dataset=name,regime=regime,arm=arm,seed=seed,selected=chosen,candidates=candidate_records,
                    targets=y['test'].tolist(),validation_targets=y['val'].tolist(),predictions=pred.tolist(),
                    error=s['binary_loss'](y['test'],pred),seconds=time.perf_counter()-t,
                    imputation_statistics=imputer.statistics_.tolist(),test_ids=meta['panels'][key]['ids']['test']))
                out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
                print(key,seed,arm,round(r['records'][-1]['error'],5),round(time.perf_counter()-t,2),flush=True)
    r['seconds']=time.perf_counter()-start
    r['summary']=s['audit_result'](r);r['status']='COMPLETE'
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r

def audit_result(r):
    """Recompute scores, selection and complete coverage before publishing ranks."""
    expected=set(itertools.product(r['tasks'],['random','temporal'],r['arms'],r['config']['seeds']))
    observed=[(z['dataset'],z['regime'],z['arm'],z['seed']) for z in r['records']]
    if len(observed)!=len(set(observed)) or set(observed)!=expected: raise ValueError('Incomplete or duplicated design')
    for key,meta in r['data']['panels'].items():
        ids=[meta['ids'][p] for p in ['train','val','test']]
        if len(set(sum(ids,[])))!=sum(map(len,ids)): raise ValueError('Overlapping/duplicate row IDs')
        if key.endswith('/temporal') and not meta['ordered_full_split']: raise ValueError('Unordered temporal partitions')
    for z in r['records']:
        meta=r['data']['panels'][z['dataset']+'/'+z['regime']]
        if z['test_ids']!=meta['ids']['test'] or z['targets']!=meta['targets']['test'] or z['validation_targets']!=meta['targets']['val']: raise ValueError('Row/target identity changed')
        if len(z['candidates'])!=2: raise ValueError('Two candidates required per arm')
        for c in z['candidates']:
            if not np.isclose(binary_loss(z['validation_targets'],c['validation_predictions']),c['validation_loss'],atol=1e-12,rtol=0): raise ValueError('Validation score changed')
            if c['best_epoch'] is not None and (c['best_epoch']!=choose_validation(c['history'])+1 or not np.isclose(c['validation_loss'],min(c['history']),atol=1e-7)): raise ValueError('Checkpoint selection changed')
        if z['selected']!=choose_validation([c['validation_loss'] for c in z['candidates']]): raise ValueError('Wrong validation selection')
        if not np.isclose(binary_loss(z['targets'],z['predictions']),z['error'],atol=1e-12,rtol=0): raise ValueError('Test score changed')
    index=dict(zip(observed,r['records']));summary={}
    for regime in ['random','temporal']:
        v=np.array([[[index[d,regime,a,k]['error'] for k in r['config']['seeds']] for a in r['arms']] for d in r['tasks']])
        means=v.mean(2);ranks=np.array([rankdata(row) for row in means])
        summary[regime]=dict(means=means.tolist(),sample_sd=v.std(2,ddof=1).tolist() if v.shape[2]>1 else None,
            ranks=ranks.tolist(),mean_ranks=dict(zip(r['arms'],ranks.mean(0).tolist())),
            inference='Two underlying datasets only; no population significance claim. Seeds are conditional repetitions.')
    return summary

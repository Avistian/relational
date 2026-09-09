"""Frozen-budget real-data checkpoint runner for L060/L064/L070.

Common splits and metric; model-specific train-only preprocessing. Test labels
never reach fitting or candidate selection. All selected predictions are retained.
"""
import copy
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from relkit.data import load_tier_a,CACHE,SPECS
from relkit.tabm import MLP,TabM
from relkit.realmlp import RealMLPS,RobustSmooth,coslog4
from relkit.benchmark_core import choose_validation,validate_partitions,paired_summary

ROOT=Path(__file__).resolve().parents[1]
PRESETS={'smoke':dict(cap=240,eval_cap=60,epochs=6,trees=30,seeds=[0],candidates=1),
         'lab':dict(cap=900,eval_cap=180,epochs=24,trees=100,seeds=[0,1,2],candidates=2),
         'closer':dict(cap=4000,eval_cap=600,epochs=128,trees=800,seeds=[0,1,2],candidates=2)}


def random_task(name,cap=900,seed=60):
    # Keep provenance alongside values, with exact row identities before reset_index.
    if name=='breast_cancer':
        x,y=load_breast_cancer(return_X_y=True,as_frame=True)
        digest=hashlib.sha256(pd.util.hash_pandas_object(x,index=True).values.tobytes()+y.to_numpy().tobytes()).hexdigest()
        source='sklearn Wisconsin Diagnostic Breast Cancer'
    else:
        x,y=load_tier_a(name);source=f'OpenML {SPECS[name]["openml_id"]}'
        digest=hashlib.sha256((CACHE/f'{name}.parquet').read_bytes()).hexdigest()
    y=np.asarray(y,dtype='int64');ids=np.arange(len(y))
    if len(ids)>cap:ids=np.sort(np.random.default_rng(seed).choice(ids,cap,replace=False))
    train,test=train_test_split(ids,test_size=.2,random_state=seed,stratify=y[ids])
    train,val=train_test_split(train,test_size=.25,random_state=seed+1,stratify=y[train])
    parts=dict(train=train,val=val,test=test);validate_partitions(parts)
    audit=dict(source=source,data_sha256=digest,split_seed=seed,row_sampling_seed=seed,
               ids={s:v.tolist() for s,v in parts.items()},regression=False)
    return {s:x.iloc[ix].reset_index(drop=True) for s,ix in parts.items()},{s:y[ix] for s,ix in parts.items()},False,audit


def encode_train(raw):
    num=list(raw['train'].select_dtypes(include='number').columns)
    cat=[c for c in raw['train'] if c not in num]
    pieces=[]
    if num:pieces.append(('num',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True)),num))
    if cat:
        raw={s:v.copy() for s,v in raw.items()}
        for v in raw.values():v[cat]=v[cat].astype(str)
        pieces.append(('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat))
    prep=ColumnTransformer(pieces,sparse_threshold=0)
    train=prep.fit_transform(raw['train']).astype('float32')
    return dict(train=train,**{s:prep.transform(raw[s]).astype('float32') for s in ['val','test']})


def error(y,p,regression):
    return float(np.sqrt(np.mean((y-p)**2))) if regression else float(log_loss(y,p,labels=[0,1]))


def fit_candidate(arm,train,ytrain,val,yval,regression,seed,candidate,cfg):
    """Returns a frozen predictor, validation error and selected epoch; no test inputs."""
    if arm in ['XGBoost','CatBoost']:
        if arm=='XGBoost':
            cls=XGBRegressor if regression else XGBClassifier
            model=cls(n_estimators=cfg['trees'],max_depth=[3,6][candidate],learning_rate=.05,
                      n_jobs=1,random_state=seed,subsample=.8,colsample_bytree=.8,tree_method='hist')
        else:
            cls=CatBoostRegressor if regression else CatBoostClassifier
            model=cls(iterations=cfg['trees'],depth=[4,6][candidate],learning_rate=.05,
                      thread_count=1,random_seed=seed,verbose=False,allow_writing_files=False)
        model.fit(train,ytrain)
        pred=model.predict if regression else lambda x:model.predict_proba(x)[:,1]
        return pred,error(yval,pred(val),regression),None
    if arm in ['TabPFN-v2','TabICL-v1.1']:
        if regression:raise ValueError('This checkpoint uses binary pretrained classifiers only')
        if arm=='TabPFN-v2':
            from tabpfn import TabPFNClassifier
            model=TabPFNClassifier(n_estimators=1,device='cpu',random_state=seed,
                model_path=str(ROOT/'data/cache/foundation/tabpfn-v2.ckpt'),n_jobs=1)
        else:
            from tabicl import TabICLClassifier
            model=TabICLClassifier(n_estimators=1,device='cpu',random_state=seed,n_jobs=1,use_amp=False,
                model_path=str(ROOT/'data/cache/foundation/tabicl-v1.1.ckpt'),
                checkpoint_version='tabicl-classifier-v1.1-0506.ckpt',allow_auto_download=False)
        model.fit(train,ytrain)
        def pred(x):
            return np.concatenate([model.predict_proba(z)[:,1] for z in np.array_split(x,max(1,int(np.ceil(len(x)/128))))])
        return pred,error(yval,pred(val),False),None
    torch.manual_seed(seed)
    prep=RobustSmooth().fit(train) if arm=='RealMLP-TD-S' else StandardScaler().fit(train)
    tr=torch.tensor(prep.transform(train),dtype=torch.float32)
    va=torch.tensor(prep.transform(val),dtype=torch.float32)
    mu,sd=(float(ytrain.mean()),max(float(ytrain.std()),1e-8)) if regression else (0.,1.)
    yt=torch.tensor((ytrain-mu)/sd,dtype=torch.float32 if regression else torch.long)
    if arm=='RealMLP-TD-S':
        model=RealMLPS(tr.shape[1],width=64,regression=regression)
        opt=torch.optim.Adam(model.parameter_groups(),betas=(.9,.95),eps=1e-8,weight_decay=0.)
    else:
        cls=TabM if arm=='TabM-mini' else MLP
        extra=dict(k=8,arch='mini',seed=seed) if arm=='TabM-mini' else {}
        model=cls(din=tr.shape[1],width=64,depth=3,dropout=.1,regression=regression,**extra)
        opt=torch.optim.AdamW(model.parameters(),lr=[.001,.003][candidate],weight_decay=1e-4)
    def prediction(z):
        model.eval()
        with torch.no_grad():
            out=model.predict(z) if arm=='TabM-mini' else model(z)
            if arm!='TabM-mini':out=out[:,0] if regression else out.softmax(-1)
        p=out.numpy();return p*sd+mu if regression else p[:,1]
    best=float('inf');state=None;best_epoch=None;steps=max(1,int(np.ceil(len(tr)/128)))
    for epoch in range(cfg['epochs']):
        model.train()
        for batch,idx in enumerate(torch.randperm(len(tr)).split(128)):
            if arm=='RealMLP-TD-S':
                lr=(.07 if regression else .04)*[1.,.5][candidate]*coslog4((epoch*steps+batch)/(cfg['epochs']*steps))
                for group in opt.param_groups:group['lr']=lr*group['factor']
            if arm=='TabM-mini':loss=model.member_losses(tr[idx],yt[idx])
            else:
                out=model(tr[idx]);loss=nn.functional.mse_loss(out[:,0],yt[idx]) if regression else nn.functional.cross_entropy(out,yt[idx],label_smoothing=.1 if arm=="RealMLP-TD-S" else 0.)
            opt.zero_grad();loss.backward();opt.step()
        score=error(yval,prediction(va),regression)
        if score<=best:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(state)
    return lambda x:prediction(torch.tensor(prep.transform(x),dtype=torch.float32)),best,best_epoch


def run_checkpoint(lesson=60,preset='lab',output=None):
    torch.set_num_threads(1);cfg=PRESETS[preset].copy()
    if lesson==60:
        names=['diabetes','blood_transfusion','kc1','phoneme','credit_g','churn','bank_marketing','adult']
        arms=['XGBoost','CatBoost','MLP','RealMLP-TD-S','TabM-mini']
    else:
        names=['diabetes','blood_transfusion','kc1','phoneme','breast_cancer']
        arms=['XGBoost','TabM-mini','TabPFN-v2','TabICL-v1.1']
        cfg['cap']=min(cfg['cap'],600) if preset=='lab' else cfg['cap']
    if preset=='smoke':names=names[:1]
    result=dict(lesson=lesson,preset=preset,config=cfg,records=[],datasets={},verdict='INCOMPARABLE',
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn','catboost','xgboost']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Numeric/one-hot inputs for all arms; fixed small recipes; no published benchmark reproduction')
    out=Path(output or ROOT/f'_verify_l{lesson:03}_results.json')
    # Resume is allowed only for exactly matching operator/config/environment identities.
    if out.exists():
        old=json.loads(out.read_text())
        if any(old.get(k)!=result[k] for k in ['lesson','preset','config','versions','source_sha256']):
            raise ValueError(f'{out}: incompatible run identity; choose a new output path')
        result=old
    tasks=[(name,'random') for name in names]
    if lesson==60 and preset!='smoke':
        tasks += [(name,split) for name in ['ecom-offers','homesite-insurance','sberbank-housing'] for split in ['random','temporal']]
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        for name,split in tasks:
            key=name+'/'+split
            if name in names:raw,y,reg,audit=random_task(name,cfg['cap'],60 if lesson==60 else 70)
            else:
                from relkit.temporal_experiment import load_release
                raw,y,reg,audit=load_release(name,split,train_cap=int(cfg['cap']*.6),eval_cap=cfg['eval_cap'],root=ROOT/'data/cache/l055')
                raw={s:pd.DataFrame(v) for s,v in raw.items()}
            x=encode_train(raw);result['datasets'][key]=audit
            for seed in cfg['seeds']:
                for arm in arms:
                    if any(r['dataset']==key and r['seed']==seed and r['arm']==arm for r in result['records']):continue
                    begin=time.perf_counter();candidates=[]
                    for candidate in range(1 if arm.startswith(('TabPFN','TabICL')) else cfg['candidates']):
                        candidates.append(fit_candidate(arm,x['train'],y['train'],x['val'],y['val'],reg,seed,candidate,cfg))
                    selected=choose_validation([c[1] for c in candidates]);predict,valid,epoch=candidates[selected]
                    fit_seconds=time.perf_counter()-begin;tick=time.perf_counter();pred=predict(x['test']);predict_seconds=time.perf_counter()-tick
                    record=dict(dataset=key,arm=arm,seed=seed,error=error(y['test'],pred,reg),metric='RMSE' if reg else 'log_loss',
                        seconds=fit_seconds+predict_seconds,fit_selection_seconds=fit_seconds,predict_seconds=predict_seconds,
                        validation_errors=[c[1] for c in candidates],selected=selected,epoch=epoch,
                        predictions=pred.tolist(),targets=y['test'].tolist())
                    result['records'].append(record);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                    print(key,seed,arm,round(record['error'],5),round(record['seconds'],2),flush=True)
    result['summary']={}
    for split in ['random','temporal']:
        records=[r for r in result['records'] if r['dataset'].endswith('/'+split)]
        if records:result['summary'][split]=paired_summary(records)
    result['last_invocation_seconds']=time.perf_counter()-start
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

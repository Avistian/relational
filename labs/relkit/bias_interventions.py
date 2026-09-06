"""L051: Grinsztajn §5 interventions; independent local training protocol.

The source-audited classifier smooths training targets, then thresholds at > .5.
We use robust training covariance without the upstream silent fallback, and record
an explicit eigenvalue floor. Standard Gaussian added columns follow Fig.5's prose;
the pinned implementation instead matches training-column means and IQRs.
"""
from __future__ import annotations
import copy
import hashlib
import json
import time
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import special_ortho_group, t, rankdata, friedmanchisquare, studentized_range
from scipy.spatial.distance import cdist
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import MinCovDet
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import fetch_openml
from xgboost import XGBClassifier
from relkit.checkpoint import CheckpointFT, CheckpointMLP

DATASETS={'electricity':44120,'MagicTelescope':44125,'bank-marketing':44126}
PRESETS={'smoke':dict(max_rows=500,epochs=3,trees=25,seeds=[0]),
         'lab':dict(max_rows=1800,epochs=20,trees=120,seeds=[0,1,2]),
         'closer':dict(max_rows=6000,epochs=50,trees=400,seeds=[0,1,2]),
         'paper':dict(max_rows=20000,epochs=100,trees=1000,seeds=[0,1,2,3,4])}
CONDITIONS=['original','rotated','noise','top5','smoothed']
MODELS=['MLP','FT-T','XGBoost']

def smooth_targets(x,y,h,covariance):
    """Appendix A.4.1: normalized Gaussian weights, including self, then > .5.

    h is a lengthscale: kernel covariance is h**2 * covariance.
    Chunks limit memory, not neighbor count. Caller supplies training-only inputs.
    """
    x,y=np.asarray(x,float),np.asarray(y,float)
    if h<0: raise ValueError('Lengthscale must be nonnegative')
    if h==0:return y.copy(),y.astype(int).copy()
    precision=np.linalg.inv(covariance)
    probability=np.empty(len(y))
    for start in range(0,len(x),256):
        distances=cdist(x[start:start+256],x,'mahalanobis',VI=precision)**2
        weights=np.exp(-distances/(2*h*h))
        probability[start:start+256]=(weights@y)/weights.sum(axis=1)
    return probability,(probability>.5).astype(int)

def rotate_splits(splits,rotation):
    """§5.4: the same invertible coordinate change on train, validation and test."""
    r=np.asarray(rotation)
    if not np.allclose(r.T@r,np.eye(len(r)),atol=1e-7):
        raise ValueError('Rotation must be orthogonal')
    return [np.asarray(x)@r for x in splits]

def add_noise_features(splits,count,seed):
    """Fig.5 prose variant: independent N(0,1) coordinates; no label input."""
    rng=np.random.default_rng(seed)
    return [np.column_stack([x,rng.standard_normal((len(x),count))]) for x in splits]

def paired_effect(changed,baseline):
    """Change in accuracy; uncertainty is conditional on this split/intervention."""
    delta=np.asarray(changed)-np.asarray(baseline)
    mean=float(delta.mean())
    if len(delta)<2:return dict(mean=mean,sd=None,ci95=None)
    sd=float(delta.std(ddof=1));half=float(t.ppf(.975,len(delta)-1)*sd/len(delta)**.5)
    return dict(mean=mean,sd=sd,ci95=[mean-half,mean+half])

def load_task(name,cache):
    path=Path(cache)/f'{name}.parquet'
    if not path.exists():
        data=fetch_openml(data_id=DATASETS[name],as_frame=True,parser='auto',data_home=str(Path(cache)/'openml'))
        frame=data.data.copy();frame['__target__']=data.target
        path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False)
    frame=pd.read_parquet(path);raw=frame.pop('__target__').astype(str)
    classes=sorted(raw.unique())
    if len(classes)!=2:raise ValueError('This lesson requires binary targets')
    x=frame.to_numpy(dtype=float);y=(raw==classes[1]).to_numpy(dtype=int)
    if not np.isfinite(x).all():raise ValueError('Released numeric data expected to be finite')
    return x,y,dict(openml_id=DATASETS[name],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),classes=classes,full_rows=len(y))

def prepare_task(name,cache,max_rows,split_seed=51):
    x,y,metadata=load_task(name,cache)
    selected=np.arange(len(y))
    if len(y)>max_rows:
        selected=np.sort(np.random.default_rng(51).choice(len(y),max_rows,replace=False))
    x,y=x[selected],y[selected]
    tr,rest=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=split_seed)
    va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=split_seed)
    transformer=QuantileTransformer(n_quantiles=min(1000,len(tr)),output_distribution='normal',random_state=51,subsample=None)
    original=[transformer.fit_transform(x[tr]),transformer.transform(x[va]),transformer.transform(x[te])]
    labels=[y[tr],y[va],y[te]]
    # Labels in va/te never enter feature selection, covariance or smoothing.
    ranking=RandomForestClassifier(n_estimators=100,random_state=51,n_jobs=1).fit(original[0],labels[0]).feature_importances_
    keep=np.argsort(ranking,kind='stable')[-min(5,x.shape[1]):]
    top=[a[:,keep] for a in original]
    cov=MinCovDet(random_state=51).fit(top[0]).covariance_
    vals,vecs=np.linalg.eigh(cov);floor=max(float(vals.max())*1e-8,1e-10)
    cov=(vecs*np.maximum(vals,floor))@vecs.T
    probability,smoothed=smooth_targets(top[0],labels[0],.5,cov)
    rotation=special_ortho_group.rvs(x.shape[1],random_state=51)
    states={'original':(original,labels[0]),'rotated':(rotate_splits(original,rotation),labels[0]),
            'noise':(add_noise_features(original,2*x.shape[1],51),labels[0]),
            'top5':(top,labels[0]),'smoothed':(top,smoothed)}
    metadata.update(selected_rows=selected.tolist(),split_rows=[selected[i].tolist() for i in [tr,va,te]],
                    dimensions=x.shape[1],top5=keep.tolist(),covariance=cov.tolist(),eigenvalue_floor=floor,
                    rotation=rotation.tolist(),smoothed_probability=probability.tolist(),
                    changed_training_labels=int(np.sum(smoothed!=labels[0])),test_y=labels[2].tolist(),
                    smoothing_h=.5,split_seed=split_seed,transform_seed=51)
    return states,labels,metadata

def fit_arm(arm,x,y,yvalid,seed,epochs,trees,device='cpu'):
    """Visible local protocol: fixed HPs, validation early stopping, no test selection.

    All conditions use identical caps. These are sensitivity measurements at fixed
    training recipes, not equal-compute optimized model-family rankings.
    """
    torch.manual_seed(seed)
    if arm=='XGBoost':
        model=XGBClassifier(n_estimators=trees,max_depth=4,learning_rate=.05,subsample=.8,
                            colsample_bytree=1.,n_jobs=1,tree_method='hist',random_state=seed,
                            eval_metric='logloss',early_stopping_rounds=12)
        model.fit(x[0],y,eval_set=[(x[1],yvalid)],verbose=False)
        return model.predict_proba(x[2])[:,1],dict(best_epoch=int(model.best_iteration)+1,best_validation_loss=float(model.best_score))
    model=(CheckpointMLP(x[0].shape[1],d=64) if arm=='MLP' else
           CheckpointFT(x[0].shape[1],d=32,layers=2,heads=4)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    xx=torch.as_tensor(x[0],dtype=torch.float32,device=device)
    yy=torch.as_tensor(y,dtype=torch.float32,device=device)
    xv=torch.as_tensor(x[1],dtype=torch.float32,device=device)
    yv=torch.as_tensor(yvalid,dtype=torch.float32,device=device)
    generator=torch.Generator().manual_seed(seed)
    best=float('inf');best_epoch=0;state=None
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(xx),generator=generator).split(256):
            optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(model(xx[idx.to(device)]),yy[idx.to(device)])
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():value=float(F.binary_cross_entropy_with_logits(model(xv),yv))
        if value<best:
            best=value;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict())
        elif epoch+1-best_epoch>=12:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        p=torch.cat([model(torch.as_tensor(a,dtype=torch.float32,device=device)).sigmoid().cpu() for a in np.array_split(x[2],max(1,int(np.ceil(len(x[2])/512))))]).numpy()
    return p,dict(best_epoch=best_epoch,best_validation_loss=best)

def rank_summary(datasets,condition):
    scores=np.array([[np.mean([r['accuracy'] for r in row['runs'][condition][m]]) for m in MODELS] for row in datasets.values()])
    ranks=np.array([rankdata(-r) for r in scores]);n,k=scores.shape
    p=float(friedmanchisquare(*scores.T).pvalue) if n>1 else None
    cd=float(studentized_range.ppf(.95,k,np.inf)/2**.5*(k*(k+1)/(6*n))**.5)
    return dict(mean_ranks=dict(zip(MODELS,ranks.mean(0).tolist())),friedman_p=p,cd=cd,N=n)

def run_experiment(preset='lab',names=None,cache=None,device='cpu',progress=True):
    cfg=PRESETS[preset].copy();names=list(DATASETS) if names is None else list(names)
    cache=Path(cache or Path(__file__).resolve().parents[1]/'data/cache/l051')
    torch.set_num_threads(1);start=time.perf_counter()
    result=dict(preset=preset,config=cfg,protocol='L051 fixed-recipe sensitivity v1',verdict='INCOMPARABLE',
                versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','xgboost','pandas']},
                device=device,datasets={})
    for name in names:
        states,labels,row=prepare_task(name,cache,cfg['max_rows']);row['runs']={}
        for condition,(xs,train_y) in states.items():
            row['runs'][condition]={}
            for arm in MODELS:
                runs=[]
                for seed in cfg['seeds']:
                    p,log=fit_arm(arm,xs,train_y,labels[1],seed,cfg['epochs'],cfg['trees'],device)
                    runs.append(dict(seed=seed,accuracy=float(accuracy_score(labels[2],p>=.5)),
                                     auc=float(roc_auc_score(labels[2],p)),probability=p.tolist(),**log))
                row['runs'][condition][arm]=runs
            if progress:print(name,condition,{m:round(np.mean([r['accuracy'] for r in row['runs'][condition][m]]),4) for m in MODELS},flush=True)
        row['effects']={c:{m:paired_effect([r['accuracy'] for r in row['runs'][c][m]],
                            [r['accuracy'] for r in row['runs']['top5' if c=='smoothed' else 'original'][m]])
                            for m in MODELS} for c in ['rotated','noise','smoothed']}
        result['datasets'][name]=row
    result['statistics']={c:rank_summary(result['datasets'],c) for c in CONDITIONS}
    result['elapsed_s']=time.perf_counter()-start
    return result

"""L057 OOF cross-family stack. Fixed recipes avoid held-out early-stopping leakage."""
import hashlib, importlib.metadata, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import rankdata, friedmanchisquare, studentized_range, t
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from relkit.tabm import TabM
from relkit.data import load_tier_a, SPECS, CACHE
from relkit.cross_ensemble import train_indices, write_oof, blend, binary_loss, greedy_select

PRESETS={
 'smoke':dict(cap=360,folds=3,epochs=8,trees=40,k=4,width=32,steps=20,tfm_estimators=1),
 'lab':dict(cap=750,folds=3,epochs=32,trees=120,k=8,width=48,steps=40,tfm_estimators=1),
 'closer':dict(cap=2000,folds=5,epochs=64,trees=300,k=32,width=128,steps=40,tfm_estimators=4),
}
CHECKPOINT='tabicl-classifier-v1.1-0506.ckpt'


def load_binary(name, cap, seed=57):
    x,y=load_tier_a(name);x=x.to_numpy(dtype='float32');y=y.to_numpy(dtype='int64')
    if not np.array_equal(np.unique(y),[0,1]):raise ValueError('Binary datasets only')
    ids=np.arange(len(y));rng=np.random.default_rng(seed)
    ids=rng.permutation(ids)[:min(cap,len(ids))]
    dev,test=train_test_split(ids,test_size=.25,random_state=seed,stratify=y[ids])
    return dict(x=x[dev],y=y[dev],xt=x[test],yt=y[test],dev_ids=dev.tolist(),test_ids=test.tolist(),
                data_hash=hashlib.sha256((CACHE/f'{name}.parquet').read_bytes()).hexdigest(),
                openml_id=SPECS[name]['openml_id'])


def fit_predict(kind, x, y, queries, seed, cfg, device='cpu', checkpoint=None):
    """Only fitting rows enter preprocessing/context/training. No validation labels."""
    if kind=='TabICL':
        from tabicl import TabICLClassifier
        model=TabICLClassifier(n_estimators=cfg['tfm_estimators'],device=device,random_state=seed,
              n_jobs=1,model_path=checkpoint,checkpoint_version=CHECKPOINT,allow_auto_download=checkpoint is None,
              use_amp=False)
        model.fit(x,y)
        return [model.predict_proba(q)[:,list(model.classes_).index(1)] for q in queries]
    prep=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    z=prep.fit_transform(x).astype('float32');qs=[prep.transform(q).astype('float32') for q in queries]
    if kind=='XGB':
        model=XGBClassifier(n_estimators=cfg['trees'],max_depth=3,learning_rate=.05,
            subsample=.8,colsample_bytree=.8,tree_method='hist',n_jobs=1,random_state=seed,eval_metric='logloss')
        model.fit(z,y)
        return [model.predict_proba(q)[:,1] for q in qs]
    torch.manual_seed(seed)
    model=TabM(z.shape[1],k=cfg['k'],width=cfg['width'],depth=3,dropout=.1,arch='mini',seed=seed).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=0.)
    xt=torch.as_tensor(z,device=device);yt=torch.as_tensor(y,device=device)
    for _ in range(cfg['epochs']):
        model.train()
        for idx in torch.randperm(len(z),device=device).split(128):
            loss=model.member_losses(xt[idx],yt[idx]);opt.zero_grad();loss.backward();opt.step()
    model.eval()
    with torch.no_grad():return [model.predict(torch.as_tensor(q,device=device))[:,1].cpu().numpy() for q in qs]


def oof_library(data, families, seed, cfg, device='cpu', checkpoint=None):
    """Each dev row predicted once; test predictions averaged over the same fold models."""
    x,y=data['x'],data['y'];fold_ids=np.full(len(y),-1,int)
    for f,(_,held) in enumerate(StratifiedKFold(cfg['folds'],shuffle=True,random_state=57).split(x,y)):
        fold_ids[held]=f
    oof=np.full((len(y),len(families)),np.nan);seen=np.zeros(len(y),int)
    test_folds=[];audit=[]
    for f in range(cfg['folds']):
        tr=train_indices(fold_ids,f);held=np.flatnonzero(fold_ids==f)
        assert not set(tr)&set(held)
        fold_pred=[];fold_test=[]
        for kind in families:
            a,b=fit_predict(kind,x[tr],y[tr],[x[held],data['xt']],seed*100+f,cfg,device,checkpoint)
            fold_pred.append(a);fold_test.append(b)
        write_oof(oof,seen,held,np.column_stack(fold_pred))
        test_folds.append(np.column_stack(fold_test))
        audit.append(dict(fold=f,fit_rows=tr.tolist(),held_rows=held.tolist()))
    if not np.all(seen==1) or not np.isfinite(oof).all():raise ValueError('OOF coverage incomplete')
    return oof,np.mean(test_folds,axis=0),audit


def evaluate_library(oof, test, y, yt, families, steps=40):
    """Freeze all choices using OOF before reading test labels for scores."""
    weights,trace=greedy_select(oof,y,steps)
    chosen=int(np.argmin([binary_loss(y,oof[:,j]) for j in range(len(families))]))
    predictions={name:test[:,j] for j,name in enumerate(families)}
    predictions['OOF-best']=test[:,chosen]
    predictions['Uniform']=blend(test,np.ones(len(families))/len(families))
    predictions['Greedy-stack']=blend(test,weights)
    errors={a:binary_loss(yt,p) for a,p in predictions.items()}
    return dict(errors=errors,weights=weights.tolist(),selected=families[chosen],trace=trace,
      oof_losses={a:binary_loss(y,oof[:,j]) for j,a in enumerate(families)},
      residual_correlation=np.corrcoef((oof-y[:,None]).T).tolist(),
      paired_gap=errors['Greedy-stack']-errors['OOF-best'])


def summarize(rows):
    names=list(dict.fromkeys(r['dataset'] for r in rows));arms=list(rows[0]['errors']);table={}
    for name in names:
        rs=[r for r in rows if r['dataset']==name];table[name]={}
        for a in arms+['paired_gap']:
            v=np.array([r['errors'][a] if a in arms else r[a] for r in rs]);n=len(v)
            sd=float(v.std(ddof=1)) if n>1 else None
            half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
            table[name][a]=dict(mean=float(v.mean()),sd=sd,ci95=None if n<2 else [float(v.mean()-half),float(v.mean()+half)])
    means=np.array([[table[n][a]['mean'] for a in arms] for n in names]);ranks=rankdata(means,axis=1)
    stat=friedmanchisquare(*means.T) if len(names)>1 else None
    cd=float(studentized_range.ppf(.95,len(arms),np.inf)/np.sqrt(2)*np.sqrt(len(arms)*(len(arms)+1)/(6*len(names))))
    return dict(table=table,arms=arms,mean_ranks=dict(zip(arms,ranks.mean(0).tolist())),
       per_dataset_ranks=ranks.tolist(),friedman_p=float(stat.pvalue) if stat else None,nemenyi_cd=cd,
       uncertainty='Paired training-seed t intervals conditional on fixed rows and folds; not population confidence.')


def run_suite(preset='lab',names=('diabetes','blood_transfusion','phoneme'),seeds=(0,1,2),
              include_tfm=False,device='cpu',checkpoint=None,output_dir='data/cache/l057',on_progress=True):
    cfg=PRESETS[preset];families=['XGB','TabM']+(['TabICL'] if include_tfm else [])
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    start=time.perf_counter();rows=[];datasets={}
    with threadpool_limits(limits=1):
        for name in names:
            data=load_binary(name,cfg['cap']);datasets[name]={k:v for k,v in data.items() if k not in ('x','y','xt','yt')}
            for seed in seeds:
                oof,test,audit=oof_library(data,families,seed,cfg,device,checkpoint)
                r=evaluate_library(oof,test,data['y'],data['yt'],families,cfg['steps']);r.update(dataset=name,seed=seed)
                rows.append(r)
                np.savez_compressed(root/f'{name}-{seed}.npz',oof=oof,test=test,y=data['y'],yt=data['yt'])
                (root/f'{name}-{seed}-folds.json').write_text(__import__('json').dumps(audit))
                if on_progress:print(name,seed,r['errors'],r['weights'],flush=True)
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','xgboost']+(['tabicl'] if include_tfm else [])}
    return dict(preset=preset,config=cfg,families=families,seeds=list(seeds),split_seed=57,fold_seed=57,
        datasets=datasets,rows=rows,summary=summarize(rows),versions=versions,seconds=time.perf_counter()-start,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
        verdict='INCOMPARABLE',scope='OOF stack with fixed recipes; not TabArena Figure 6 reproduction')

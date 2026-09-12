"""L069: visible open-environment operators and full historical-v2 diagnostics.

Original evaluation source is attributed in sources/l069-v2. This independently
written evaluator fixes class aggregation and records continuous and binary
novelty scores separately. Frozen L064 full checkpoint implementation is reused.
"""
import hashlib,importlib.metadata,inspect,itertools,json,math,time,types,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,average_precision_score
from xgboost import XGBClassifier
from relkit.tabpfn_l064_v2 import TabPFNv2,MODEL_CONFIG,CHECKPOINT_SHA,ensure_checkpoint,load_pretrained,predict_numeric
SOURCE_COMMIT='744c010457f68284faa7ae6ded793a8b3f3e03a4'
SOURCE_URL='https://raw.githubusercontent.com/LAMDA-NeSy/Evaluation-on-Tabular-Model-in-Open-Environments/'+SOURCE_COMMIT+'/'
DATA_SHA={'cmc':'f1954d85d813fb0586cbc27cfe407e40e238ea0b39a3cf5c652694c1f4225d95','winequality-red':'65fc57ebcf15bf3264b2e00212539a2c0840f1d6abf6ae9e8f018d58030b4b29','winequality-white':'b8f4937cecb06e57ba75f5b851e1d0376b3d507d1743d5409cd20bd920120f93','iris':'b5bb2427eb7ff7ca54f3d1ad6fe52610caf38b1af3a6bfaf965dec5d63af2d6d'}
PROTOCOL=dict(novelty='all labels; balanced novel/known disjoint test; all remaining known rows context',features='80/20 nonstratified split; nested label-blind feature permutation; train means; fixed context',seeds=[42,2023,789],levels=[0,.2,.4,.6,.8,1.],model_seed=0,temperature=.9,xgboost=dict(n_estimators=100,max_depth=6,learning_rate=.3,subsample=1.,colsample_bytree=1.,n_jobs=1,tree_method='hist'),epsilon=1e-12,quantization='retain source CSV floating-point values; no source integer truncation',aggregation='classes equally within dataset/seed; seeds within dataset; datasets equally')
PRESETS69={'smoke':dict(novelty=['iris'],features=['iris'],seeds=[42],cap=120),'lab':dict(novelty=['cmc'],features=['iris'],seeds=[42],cap=None),'closer':dict(novelty=['cmc','winequality-red','winequality-white'],features=['iris','cmc','winequality-red'],seeds=[42,2023,789],cap=None)}

def novel_split(y,held_label,seed):
    """App E.2: every novel row plus equal held-out known rows; context disjoint."""
    y=np.asarray(y);novel=np.flatnonzero(y==held_label);known=np.flatnonzero(y!=held_label)
    if not 0<len(novel)<len(known):raise ValueError('Balanced source design requires fewer novel than known rows')
    strata=y[known] if len(novel)>=len(np.unique(y[known])) else None
    selected,context=train_test_split(known,train_size=len(novel),random_state=int(seed),stratify=strata)
    return dict(context=np.asarray(context),query=np.concatenate([novel,selected]),novel=np.concatenate([np.ones(len(novel),int),np.zeros(len(selected),int)]))

def novelty_scores(probabilities,lower=.4,upper=.6):
    """Separate continuous low-confidence score from the source binary interval."""
    p=np.asarray(probabilities,float)
    if p.ndim!=2 or p.shape[1]<2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Normalized class probabilities required')
    if not 0<=lower<=upper<=1:raise ValueError('Invalid interval')
    confidence=p.max(1)
    return dict(continuous=1-confidence,interval=((confidence>=lower)&(confidence<=upper)).astype(int))

def impute_features(context,query,columns):
    """Frozen model fallback; never recompute means from corrupted test rows."""
    context=np.asarray(context,float);query=np.asarray(query,float);columns=np.asarray(columns,int)
    if context.ndim!=2 or query.ndim!=2 or context.shape[1]!=query.shape[1]:raise ValueError('Matching feature schemas required')
    if (columns<0).any() or (columns>=context.shape[1]).any():raise ValueError('Unknown feature ID')
    result=query.copy();result[:,columns]=context[:,columns].mean(0)
    return result

def all_row_metrics(y,probabilities,classes,epsilon=1e-12):
    """Keep unsupported labels: zero mass -> declared clipped diagnostic, no filtering."""
    y=np.asarray(y);p=np.asarray(probabilities,float);classes=np.asarray(classes)
    if p.shape!=(len(y),len(classes)) or len(np.unique(classes))!=len(classes) or not 0<epsilon<1:raise ValueError('Class map or epsilon invalid')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid probabilities')
    supported=np.isin(y,classes);lookup={v:i for i,v in enumerate(classes.tolist())}
    target_probability=np.array([p[i,lookup[v]] if v in lookup else 0. for i,v in enumerate(y.tolist())])
    loss=-np.log(np.maximum(target_probability,epsilon));pred=classes[p.argmax(1)]
    return dict(n=len(y),unsupported_n=int((~supported).sum()),unsupported_fraction=float((~supported).mean()),accuracy=float((pred==y).mean()),known_accuracy=float((pred[supported]==y[supported]).mean()) if supported.any() else None,clipped_log_loss=float(loss.mean()),known_log_loss=float(loss[supported].mean()) if supported.any() else None,unsupported_penalty=float(-math.log(epsilon)),epsilon=epsilon)

def dataset_summary(records):
    """Average classes first, then seeds: no row/class/seed pseudoreplication."""
    keys=['axis','dataset','arm','condition'];metrics=['accuracy','balanced_accuracy','macro_f1','auc','ap','interval_auc','interval_ap','clipped_log_loss','known_accuracy','unsupported_fraction','delta_accuracy','relative_accuracy_gap']
    groups={}
    for row in records:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    out=[]
    for key,rows in groups.items():
        result=dict(zip(keys,key));result['seeds']=sorted({r['seed'] for r in rows})
        for metric in metrics:
            means=[float(np.mean([r[metric] for r in rows if r['seed']==seed and r.get(metric) is not None])) for seed in result['seeds'] if any(r['seed']==seed and r.get(metric) is not None for r in rows)]
            if means:result.update({metric+'_mean':float(np.mean(means)),metric+'_sd':float(np.std(means,ddof=1)) if len(means)>1 else None,metric+'_seed_values':means})
        out.append(result)
    return out

def shift_task(context_x,query_x,kind):
    """Generated controlled family: fixed source rule y=1[x0>0], explicit target law."""
    cx=np.asarray(context_x,float);qx=np.asarray(query_x,float).copy()
    if cx.shape[1]!=2 or qx.shape[1]!=2:raise ValueError('Two-coordinate controlled task')
    cy=(cx[:,0]>0).astype(int)
    if kind=='covariate':qx[:,0]+=1.5
    elif kind not in ['iid','concept']:raise ValueError('Unknown intervention')
    qy=(qx[:,0]>0).astype(int)
    if kind=='concept':qy=1-qy
    return cx.copy(),cy,qx,qy


def load_dataset(root,name,cap=None,seed=42):
    if name not in DATA_SHA:raise ValueError('Unknown source-prepared dataset')
    path=Path(root)/'data/cache/l069-source'/name/(name+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():urllib.request.urlretrieve(SOURCE_URL+'dataset/'+name+'/'+name+'.csv',path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=DATA_SHA[name]:raise ValueError('Source dataset bytes changed')
    df=pd.read_csv(path);target='label' if 'label' in df else df.columns[-1]
    y=pd.Categorical(df[target]).codes.astype(int);x=df.drop(columns=target).to_numpy(dtype=np.float32);ids=np.arange(len(y))
    if not np.isfinite(x).all():raise ValueError('This declared numeric panel contains no missing values')
    if cap is not None and len(y)>cap:ids=train_test_split(ids,train_size=cap,random_state=seed,stratify=y)[0]
    return x[ids],y[ids],ids,dict(sha256=DATA_SHA[name],original_rows=len(df),used_rows=len(ids),features=list(df.drop(columns=target).columns),classes=np.unique(y).tolist(),raw_numeric=True)

def metric_record(y,p,classes):
    r=all_row_metrics(y,p,classes);pred=np.asarray(classes)[np.asarray(p).argmax(1)]
    r['balanced_accuracy']=float(balanced_accuracy_score(y,pred));r['macro_f1']=float(f1_score(y,pred,average='macro',zero_division=0))
    if r['unsupported_n']==0:
        labels=np.array([list(classes).index(v) for v in y]);r['auc']=float(roc_auc_score(labels,np.asarray(p)[:,1])) if len(classes)==2 else float(roc_auc_score(labels,p,multi_class='ovr',average='macro',labels=np.arange(len(classes))))
    return r

def stable_code(obj,seen=None):
    """Semantic bytecode graph; nested generators, defaults and global helpers included."""
    seen=set() if seen is None else seen
    if isinstance(obj,types.CodeType):return [obj.co_code.hex(),obj.co_names,[stable_code(v,seen) for v in obj.co_consts]]
    if isinstance(obj,types.FunctionType):
        if id(obj) in seen:return ['recursive',obj.__name__]
        seen.add(id(obj));names=set()
        def visit(code):
            names.update(code.co_names)
            for v in code.co_consts:
                if isinstance(v,types.CodeType):visit(v)
        visit(obj.__code__)
        dependencies={n:stable_code(obj.__globals__[n],seen) for n in sorted(names) if n in obj.__globals__ and (isinstance(obj.__globals__[n],types.FunctionType) and (obj.__globals__[n].__module__==obj.__module__ or obj.__globals__[n].__module__.startswith('relkit.')))}
        return [obj.__name__,stable_code(obj.__code__,seen),stable_code(obj.__defaults__,seen),stable_code(obj.__kwdefaults__,seen),dependencies]
    if isinstance(obj,dict):return {str(k):stable_code(v,seen) for k,v in sorted(obj.items(),key=lambda p:str(p[0]))}
    if isinstance(obj,(tuple,list)):return [stable_code(v,seen) for v in obj]
    if isinstance(obj,np.ndarray):return [str(obj.dtype),obj.shape,hashlib.sha256(obj.tobytes()).hexdigest()]
    if obj is None or isinstance(obj,(bool,int,float,str)):return obj
    return [type(obj).__module__,type(obj).__name__]

def kernel_identity(namespace,root):
    names=['novel_split','novelty_scores','impute_features','all_row_metrics','dataset_summary','shift_task','load_dataset','metric_record','run_experiment','predict_numeric']
    payload=dict(functions={n:stable_code(namespace[n]) for n in names},protocol=PROTOCOL,data=DATA_SHA,checkpoint=CHECKPOINT_SHA,model_source=hashlib.sha256((Path(root)/'relkit/tabpfn_l064_v2.py').read_bytes()).hexdigest(),versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','scikit-learn','xgboost']})
    # Visible live model classes override the imported reference in standalone notebooks.
    for name in ['TabPFNv2','PackedAttention','V2Block']:
        if name in namespace:payload[name]={k:stable_code(v) for k,v in vars(namespace[name]).items() if isinstance(v,types.FunctionType)}
    return dict(sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload=payload)

def run_experiment(root,config=None,namespace=None):
    root=Path(root);s=globals() if namespace is None else namespace;cfg=dict(PRESETS69['lab'] if config is None else config)
    if cfg.get('paper'):raise ValueError('Full original benchmark is not implemented; use the documented original route')
    torch.set_num_threads(1);start=time.perf_counter();identity=s['kernel_identity'](s,root)
    model,_=s['load_pretrained'](s['ensure_checkpoint'](root),s['TabPFNv2'](**MODEL_CONFIG));records=[];datasets={};splits=[]
    def fit_predict(cx,cy,qx,arm,seed):
        classes,encoded=np.unique(cy,return_inverse=True)
        if arm=='v2':p,info=s['predict_numeric'](model,cx,cy,qx,seed=PROTOCOL['model_seed'],temperature=PROTOCOL['temperature']);return p.numpy(),info['classes']
        estimator=XGBClassifier(**PROTOCOL['xgboost'],random_state=seed);estimator.fit(cx,encoded);return estimator.predict_proba(qx),classes.tolist()
    for axis in ['novelty','features']:
        for dataset in cfg[axis]:
            for seed in cfg['seeds']:
                x,y,row_ids,metadata=s['load_dataset'](root,dataset,cfg.get('cap'),seed);datasets[dataset]=metadata
                if axis=='novelty':
                    for held in np.unique(y):
                        split=s['novel_split'](y,int(held),seed);context=split['context'];query=split['query'];novel=split['novel'];sid=f'{axis}:{dataset}:{seed}:{held}'
                        splits.append(dict(id=sid,context_ids=row_ids[context].tolist(),query_ids=row_ids[query].tolist(),targets=y[query].tolist(),novel=novel.tolist(),held_label=int(held)))
                        for arm in ['v2','xgboost']:
                            p,classes=fit_predict(x[context],y[context],x[query],arm,seed);scores=s['novelty_scores'](p);r=s['all_row_metrics'](y[query],p,classes)
                            r.update(axis=axis,dataset=dataset,seed=seed,arm=arm,condition='leave-one-class-out',held_label=int(held),split_id=sid,classes=classes,probabilities=p.tolist(),auc=float(roc_auc_score(novel,scores['continuous'])),ap=float(average_precision_score(novel,scores['continuous'])),interval_auc=float(roc_auc_score(novel,scores['interval'])),interval_ap=float(average_precision_score(novel,scores['interval'])),interval_novel_rate=float(scores['interval'][novel==1].mean()),interval_known_rate=float(scores['interval'][novel==0].mean()))
                            records.append(r);print('MEASURE',axis,dataset,seed,int(held),arm,round(r['auc'],4),flush=True)
                    # Separate natural-prevalence task: pre-split all rows, remove class0 only from context.
                    train,query=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
                    held=int(np.unique(y)[0]);context=train[y[train]!=held];novel=(y[query]==held).astype(int);sid=f'natural:{dataset}:{seed}'
                    splits.append(dict(id=sid,context_ids=row_ids[context].tolist(),query_ids=row_ids[query].tolist(),targets=y[query].tolist(),novel=novel.tolist(),held_label=held,excluded_train_ids=row_ids[train[y[train]==held]].tolist()))
                    for arm in ['v2','xgboost']:
                        p,classes=fit_predict(x[context],y[context],x[query],arm,seed);scores=s['novelty_scores'](p);r=s['all_row_metrics'](y[query],p,classes)
                        r.update(axis=axis,dataset=dataset,seed=seed,arm=arm,condition='natural-prevalence-class0',held_label=held,split_id=sid,classes=classes,probabilities=p.tolist(),auc=float(roc_auc_score(novel,scores['continuous'])),ap=float(average_precision_score(novel,scores['continuous'])),interval_auc=float(roc_auc_score(novel,scores['interval'])),interval_ap=float(average_precision_score(novel,scores['interval'])))
                        records.append(r)
                else:
                    context,query=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed);sid=f'{axis}:{dataset}:{seed}'
                    splits.append(dict(id=sid,context_ids=row_ids[context].tolist(),query_ids=row_ids[query].tolist(),targets=y[query].tolist()))
                    order=np.random.default_rng(seed).permutation(x.shape[1]);base={}
                    for level in PROTOCOL['levels']:
                        columns=order[:math.floor(level*x.shape[1])];qx=s['impute_features'](x[context],x[query],columns)
                        for arm in ['v2','xgboost']:
                            p,classes=fit_predict(x[context],y[context],qx,arm,seed);r=s['metric_record'](y[query],p,classes)
                            if level==0:base[arm]=r['accuracy']
                            r.update(axis=axis,dataset=dataset,seed=seed,arm=arm,condition=f'{level:.0%}',split_id=sid,classes=classes,probabilities=p.tolist(),columns=columns.tolist(),removed_fraction=len(columns)/x.shape[1],delta_accuracy=r['accuracy']-base[arm],relative_accuracy_gap=(r['accuracy']-base[arm])/base[arm])
                            records.append(r);print('MEASURE',axis,dataset,seed,level,arm,round(r['accuracy'],4),flush=True)
    # Analytic distribution laws isolate p(x) from p(y|x), unlike a real-world drift attribution.
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);cx=rng.normal(size=(256,2));qx=rng.normal(size=(512,2))
        for kind in ['iid','covariate','concept']:
            tx,ty,q,yq=s['shift_task'](cx,qx,kind);sid=f'shift:{seed}:{kind}'
            splits.append(dict(id=sid,context_ids=list(range(256)),query_ids=list(range(256,768)),targets=yq.tolist(),context_x=tx.tolist(),context_y=ty.tolist(),query_x=q.tolist()))
            for arm in ['v2','xgboost']:
                p,classes=fit_predict(tx,ty,q,arm,seed);r=s['metric_record'](yq,p,classes)
                r.update(axis='shift',dataset='generated-threshold',seed=seed,arm=arm,condition=kind,split_id=sid,classes=classes,probabilities=p.tolist());records.append(r)
    if s['kernel_identity'](s,root)['sha256']!=identity['sha256']:raise RuntimeError('Live code changed during measurement')
    return dict(schema='l069-v2',config=cfg,protocol=PROTOCOL,kernel_identity=identity,source_commit=SOURCE_COMMIT,datasets=datasets,splits=splits,records=records,summary=s['dataset_summary'](records),seconds=time.perf_counter()-start,paper_reproduction='INCOMPARABLE: local diagnostic; original tuned seven-model four-axis benchmark NOT_RUN')

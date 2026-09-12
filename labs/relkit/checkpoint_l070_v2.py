"""L070 v2: live selection, corrected TabM, intervention, and an honest hybrid panel.

Original seven-arm JSON and original operators are immutable inputs. No pretrained
model is refit by this default runner. The TabM architecture is in tabm_v2.py.
"""
import copy,hashlib,importlib.metadata,inspect,itertools,json,math,time,types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata,friedmanchisquare,studentized_range,t
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from relkit.data import load_tier_a,CACHE,SPECS
from relkit.tabm_v2 import TabM,BatchEnsembleLinear,PackedHead,kaiming_,sign_pm1,batchensemble_linear,packed_head,member_mean_loss,ensemble_predict

DATASETS=['diabetes/random','blood_transfusion/random','kc1/random','phoneme/random','breast_cancer/random']
ARCHIVE_ARMS=['XGBoost','TabM-mini','TabPFN-v2','TabICL-v1.1','TabPFN-2.5-synthetic','TabPFN-3','TabICLv2']
ARMS=['XGBoost','TabM-mini-v2','TabPFN-v2','TabICL-v1.1','TabPFN-2.5-synthetic','TabPFN-3','TabICLv2']
PRESETS70={'smoke':dict(datasets=DATASETS[:1],seeds=[0],epochs=4,trees=30),
 'lab':dict(datasets=DATASETS,seeds=[0,1,2],epochs=24,trees=100),
 'closer':dict(datasets=DATASETS,seeds=[0,1,2],epochs=128,trees=800)}
PROTOCOL70=dict(cap=600,split_seeds=[70,71],candidate_lrs=[.001,.003],candidate_depths=[3,6],
 width=64,depth=3,k=8,dropout=.1,batch_size=128,weight_decay=1e-4,
 intervention='largest absolute train-only Pearson correlation, replace test feature with train median; frozen predictor',
 prediction='Feature erasure increases dataset-average log loss for both freshly fitted arms; signed prediction may fail',
 epoch_tie='last minimum',candidate_tie='first minimum',positive_class=1)


def binary_loss(y,p):
    """Mean negative log likelihood in nats; float64 epsilon clips exact endpoints."""
    y,p=np.asarray(y),np.asarray(p,dtype=np.float64)
    if y.ndim!=1 or not y.size or p.shape!=y.shape or not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or ((p<0)|(p>1)).any():
        raise ValueError('Require aligned nonempty binary targets and finite P(y=1) in [0,1]')
    p=np.clip(p,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def choose_candidate(errors):
    """Smallest validation loss; exact ties choose the first declared recipe."""
    errors=np.asarray(errors,dtype=float)
    if errors.ndim!=1 or not errors.size or not np.isfinite(errors).all():raise ValueError('Finite validation vector required')
    return int(np.argmin(errors))


def complete_panel(records,datasets,arms,seeds):
    """Validate declared Cartesian design; average seeds inside datasets, then rank."""
    if any(not len(v) or len(set(v))!=len(v) for v in [datasets,arms,seeds]):raise ValueError('Unique nonempty design required')
    expected=set(itertools.product(datasets,arms,seeds));observed=[(r['dataset'],r['arm'],r['seed']) for r in records]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:raise ValueError('Missing, duplicated or undeclared crossed cell')
    indexed={k:r for k,r in zip(observed,records)}
    values=np.array([[[indexed[d,a,k]['error'] for k in seeds] for a in arms] for d in datasets],dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite score')
    means=values.mean(2);ranks=np.stack([rankdata(row,method='average') for row in means]);n,m=means.shape
    p=float(friedmanchisquare(*means.T).pvalue) if n>=3 and m>=3 and np.any(np.ptp(ranks,axis=1)) else None
    return dict(datasets=list(datasets),arms=list(arms),seeds=list(seeds),values=values.tolist(),means=means.tolist(),
        sample_sd=values.std(2,ddof=1).tolist() if len(seeds)>1 else None,dataset_ranks=ranks.tolist(),
        mean_ranks=dict(zip(arms,ranks.mean(0).tolist())),friedman_p=p,
        nemenyi_cd=float(studentized_range.ppf(.95,m,np.inf)*np.sqrt(m*(m+1)/(12*n))) if m>1 else None,
        unit='dataset; seeds are conditional downstream repetitions')


def dataset_bootstrap(differences,repetitions=2000,seed=70):
    """Input [datasets,seeds] paired differences; resample whole dataset means."""
    differences=np.asarray(differences,float)
    if differences.ndim!=2 or min(differences.shape)<1 or not np.isfinite(differences).all() or repetitions<100:raise ValueError('Finite paired dataset-by-seed matrix and >=100 replicates required')
    means=differences.mean(1);n=len(means);rng=np.random.default_rng(seed)
    draws=means[rng.integers(n,size=(repetitions,n))].mean(1)
    return dict(mean=float(means.mean()),percentile95=np.quantile(draws,[.025,.975]).tolist(),units=n,
        dataset_means=means.tolist(),repetitions=repetitions,seed=seed,scope='Empirical five-dataset resampling; not population coverage or pretraining uncertainty')


def lifecycle_cost(prepare_seconds,predict_seconds,batches):
    """One preparation plus B repeats of the same measured query batch."""
    vals=np.asarray([prepare_seconds,predict_seconds,batches],float)
    if not np.isfinite(vals).all() or (vals<0).any():raise ValueError('Finite nonnegative costs and batch count required')
    return float(prepare_seconds+batches*predict_seconds)


def erase_feature(train,query,column):
    """No refit: replace one query coordinate with its training median; copy inputs."""
    train,query=np.asarray(train),np.asarray(query)
    if train.ndim!=2 or query.ndim!=2 or train.shape[1]!=query.shape[1] or not isinstance(column,(int,np.integer)) or not 0<=column<train.shape[1]:raise ValueError('Aligned feature schema and valid column required')
    out=query.astype(float,copy=True);out[:,column]=np.median(train[:,column]);return out


def stable_code(obj,seen=None):
    """Stable semantic dependency graph including nested code, defaults and closures."""
    seen=set() if seen is None else seen
    if isinstance(obj,types.CodeType):return [obj.co_code.hex(),obj.co_names,obj.co_varnames,obj.co_freevars,obj.co_cellvars,obj.co_argcount,obj.co_posonlyargcount,obj.co_kwonlyargcount,obj.co_flags,[stable_code(v,seen) for v in obj.co_consts]]
    if inspect.isclass(obj):
        if id(obj) in seen:return ['recursive-class',obj.__name__]
        seen.add(id(obj));return {k:stable_code(v,seen) for k,v in sorted(vars(obj).items()) if isinstance(v,types.FunctionType)}
    if isinstance(obj,types.FunctionType):
        if id(obj) in seen:return ['recursive',obj.__name__]
        seen.add(id(obj));names=set()
        def visit(code):
            names.update(code.co_names)
            for v in code.co_consts:
                if isinstance(v,types.CodeType):visit(v)
        visit(obj.__code__);dependencies={}
        for n in sorted(names):
            if n not in obj.__globals__:continue
            v=obj.__globals__[n]
            if isinstance(v,(types.FunctionType,type)) and (getattr(v,'__module__',None)==obj.__module__ or str(getattr(v,'__module__','')).startswith('relkit.')):dependencies[n]=stable_code(v,seen)
            elif n.isupper() and isinstance(v,(dict,list,tuple,str,int,float)):dependencies[n]=stable_code(v,seen)
        return [obj.__name__,stable_code(obj.__code__,seen),stable_code(obj.__defaults__,seen),stable_code(obj.__kwdefaults__,seen),stable_code([v.cell_contents for v in obj.__closure__ or []],seen),dependencies]
    if isinstance(obj,dict):return {str(k):stable_code(v,seen) for k,v in sorted(obj.items(),key=lambda p:str(p[0]))}
    if isinstance(obj,(tuple,list)):return [stable_code(v,seen) for v in obj]
    if isinstance(obj,np.ndarray):return [str(obj.dtype),obj.shape,hashlib.sha256(obj.tobytes()).hexdigest()]
    if isinstance(obj,Path):return ['Path',str(obj)]
    if isinstance(obj,float) and not math.isfinite(obj):return ['nonfinite',str(obj)]
    if obj is None or isinstance(obj,(bool,int,float,str)):return obj
    return [type(obj).__module__,type(obj).__name__]


def model_identity(model):
    if isinstance(model,XGBClassifier):return dict(weights=hashlib.sha256(model.get_booster().save_raw()).hexdigest(),runtime=stable_code(model.get_params()))
    h=hashlib.sha256();runtime={}
    for name,tensor in model.state_dict().items():h.update(name.encode());h.update(str((tensor.shape,tensor.dtype)).encode());h.update(tensor.detach().numpy().tobytes())
    for name,module in model.named_modules():
        if 'forward' in vars(module) or module._forward_hooks or module._forward_pre_hooks:raise RuntimeError('Untracked model hook or instance method')
        runtime[name]=dict(type=type(module).__qualname__,forward=stable_code(module.forward.__func__),settings=stable_code({k:v for k,v in vars(module).items() if not k.startswith('_')}))
    return dict(weights=h.hexdigest(),runtime_sha256=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest())


def kernel_identity(namespace,root):
    names=['binary_loss','choose_candidate','complete_panel','dataset_bootstrap','lifecycle_cost','erase_feature','load_task','fit_candidate','run_experiment','audit_archive','model_identity','TabM','BatchEnsembleLinear','PackedHead']
    payload=dict(xgboost_methods={n:stable_code(getattr(namespace['XGBClassifier'],n)) for n in ['__init__','fit','predict_proba']},functions={n:stable_code(namespace[n]) for n in names},protocol=PROTOCOL70,
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','scipy','scikit-learn','xgboost']},
        data_loader_sha256=hashlib.sha256((Path(root)/'relkit/data.py').read_bytes()).hexdigest(),
        model_file_sha256=hashlib.sha256((Path(root)/'relkit/tabm_v2.py').read_bytes()).hexdigest(),
        archive_sha256=hashlib.sha256((Path(root)/'_verify_l070_results.json').read_bytes()).hexdigest())
    return dict(sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload=payload)


def load_task(name,root):
    """Exact original 600-row cap and 70/71 splits, with raw semantic class map."""
    name=name.split('/')[0]
    if name=='breast_cancer':
        b=load_breast_cancer(as_frame=True);x,y=b.data,b.target;class_map={str(i):str(v) for i,v in enumerate(b.target_names)}
        digest=hashlib.sha256(pd.util.hash_pandas_object(x,index=True).values.tobytes()+y.to_numpy().tobytes()).hexdigest()
    else:
        x,y=load_tier_a(name);cache=Path(root)/'data/cache'/(name+'.parquet');digest=hashlib.sha256(cache.read_bytes()).hexdigest()
        f=pd.read_parquet(cache);raw=f[SPECS[name]['target']]
        class_map={str(i):v for i,v in enumerate(sorted(raw.astype(str).unique()))} if str(raw.dtype) in ['object','category'] else {str(int(v)):str(v) for v in sorted(raw.unique())}
    y=np.asarray(y,dtype='int64');ids=np.arange(len(y))
    if len(ids)>600:ids=np.sort(np.random.default_rng(70).choice(ids,600,replace=False))
    tr,te=train_test_split(ids,test_size=.2,random_state=70,stratify=y[ids]);tr,va=train_test_split(tr,test_size=.25,random_state=71,stratify=y[tr]);parts=dict(train=tr,val=va,test=te)
    if len(np.unique(np.concatenate(list(parts.values()))))!=sum(map(len,parts.values())):raise ValueError('Overlapping partitions')
    prep=SimpleImputer(strategy='median',keep_empty_features=True).fit(x.iloc[tr])
    xx={k:prep.transform(x.iloc[v]).astype('float32') for k,v in parts.items()};yy={k:y[v] for k,v in parts.items()}
    audit=dict(ids={k:v.tolist() for k,v in parts.items()},data_sha256=digest,class_map=class_map,
        feature_names=x.columns.tolist(),imputation_median=prep.statistics_.tolist(),positive_class=1)
    return xx,yy,audit


def audit_archive(root,namespace=None):
    s=globals() if namespace is None else namespace;root=Path(root);archive=json.loads((root/'_verify_l070_results.json').read_text());maximum=0.;datasets={}
    for name in DATASETS:
        x,y,audit=s['load_task'](name,root);old=archive['datasets'][name]
        if audit['ids']!=old['ids'] or audit['data_sha256']!=old['data_sha256']:raise ValueError('Historical raw data or row identity changed')
        datasets[name]=audit
        for r in [z for z in archive['records'] if z['dataset']==name]:
            if r['targets']!=y['test'].tolist():raise ValueError('Archive targets disagree with raw data positive class')
            delta=abs(s['binary_loss'](r['targets'],r['predictions'])-r['error']);maximum=max(maximum,delta)
            if delta>2e-7:raise ValueError('Saved prediction score mismatch')
            if r['selected']!=s['choose_candidate'](r['validation_errors']):raise ValueError('Historical selection differs from validation minimum')
    summary=s['complete_panel'](archive['records'],DATASETS,ARCHIVE_ARMS,[0,1,2])
    return dict(status='PASS',records=len(archive['records']),maximum_loss_delta=maximum,datasets=datasets,
        archive_sha256=hashlib.sha256((root/'_verify_l070_results.json').read_bytes()).hexdigest(),summary=summary,
        validation_probability_reconstruction='NOT_CHECKED: original JSON saves only validation scalar errors',
        pretrained_refit='NOT_RUN by this audit; separate independent source replay may establish prediction parity',
        checkpoints=json.loads((root/'_sources_foundation.json').read_text()),historical_versions=archive['versions'],current_versions=archive['current_versions'])


def fit_candidate(arm,train,ytrain,val,yval,seed,candidate,cfg,namespace):
    """No test parameter. Persist every epoch loss and selected validation probabilities."""
    s=namespace;begin=time.perf_counter()
    if arm=='XGBoost-fresh-control':
        model=s['XGBClassifier'](n_estimators=cfg['trees'],max_depth=[3,6][candidate],learning_rate=.05,n_jobs=1,
            random_state=seed,subsample=.8,colsample_bytree=.8,tree_method='hist');model.fit(train,ytrain)
        def predict(z):return model.predict_proba(z)[:,list(model.classes_).index(1)].astype(float)
        vp=predict(val);history=[s['binary_loss'](yval,vp)];epoch=None;preprocessing=None
    else:
        torch.manual_seed(seed);prep=StandardScaler().fit(train);tr=torch.tensor(prep.transform(train),dtype=torch.float32);va=torch.tensor(prep.transform(val),dtype=torch.float32);yt=torch.tensor(ytrain,dtype=torch.long)
        model=s['TabM'](din=tr.shape[1],width=64,depth=3,k=8,arch='mini',seed=seed,dropout=.1)
        opt=torch.optim.AdamW(model.parameters(),lr=[.001,.003][candidate],weight_decay=1e-4)
        best=float('inf');history=[];state=None;epoch=None
        for step in range(cfg['epochs']):
            model.train()
            for idx in torch.randperm(len(tr)).split(128):
                loss=model.member_losses(tr[idx],yt[idx]);opt.zero_grad();loss.backward();opt.step()
            model.eval()
            with torch.no_grad():p=model.predict(va)[:,1].numpy()
            error=s['binary_loss'](yval,p);history.append(error)
            if error<=best:best=error;state=copy.deepcopy(model.state_dict());epoch=step+1
        model.load_state_dict(state);model.eval()
        def predict(z):
            with torch.no_grad():return model.predict(torch.tensor(prep.transform(z),dtype=torch.float32))[:,1].numpy().astype(float)
        vp=predict(val);preprocessing=dict(mean=prep.mean_.tolist(),scale=prep.scale_.tolist())
    trace=dict(candidate=candidate,validation_predictions=vp.tolist(),validation_error=s['binary_loss'](yval,vp),epoch_losses=history,
        selected_epoch=epoch,seconds=time.perf_counter()-begin,model_identity=s['model_identity'](model),preprocessing=preprocessing)
    return predict,trace


def run_experiment(root,config=None,namespace=None,output=None):
    """Fresh bounded fits and frozen test intervention; no resume or implicit downloads."""
    root=Path(root);s=globals() if namespace is None else namespace;cfg=copy.deepcopy(PRESETS70['lab'] if config is None else config)
    if output and Path(output).exists():raise FileExistsError('Choose a fresh output; historical evidence must remain immutable')
    torch.set_num_threads(1);identity=s['kernel_identity'](s,root);start=time.perf_counter();archive=s['audit_archive'](root,s)
    result=dict(schema='l070-v2',config=cfg,protocol=PROTOCOL70,kernel_identity=identity,archive_audit=archive,datasets={},records=[],status='IN_PROGRESS',paper_reproduction='INCOMPARABLE: local fixed five-table panel, not original paper protocols')
    for name in cfg['datasets']:
        x,y,audit=s['load_task'](name,root);result['datasets'][name]=audit
        # A prespecified target-informed TRAIN statistic, never chosen from test degradation.
        centered=x['train']-x['train'].mean(0);target=y['train']-y['train'].mean();denom=np.sqrt((centered**2).sum(0)*(target**2).sum());correlation=np.divide(centered.T@target,denom,out=np.zeros(centered.shape[1]),where=denom>0);column=int(np.argmax(abs(correlation)))
        changed=s['erase_feature'](x['train'],x['test'],column)
        for seed in cfg['seeds']:
            for arm in ['XGBoost-fresh-control','TabM-mini-v2']:
                candidates=[s['fit_candidate'](arm,x['train'],y['train'],x['val'],y['val'],seed,i,cfg,s) for i in range(2)]
                selected=s['choose_candidate']([c[1]['validation_error'] for c in candidates]);predict,trace=candidates[selected]
                # First model test prediction after validation selection. Labels only enter scoring.
                tick=time.perf_counter();p=predict(x['test']);predict_seconds=time.perf_counter()-tick;damaged=predict(changed)
                fit_seconds=sum(c[1]['seconds'] for c in candidates)
                row=dict(dataset=name,arm=arm,seed=seed,error=s['binary_loss'](y['test'],p),predictions=p.tolist(),targets=y['test'].tolist(),test_ids=audit['ids']['test'],
                    validation_ids=audit['ids']['val'],validation_targets=y['val'].tolist(),candidates=[c[1] for c in candidates],selected=selected,
                    fit_selection_seconds=fit_seconds,predict_seconds=predict_seconds,seconds=fit_seconds+predict_seconds,
                    lifecycle_100_batches_seconds=s['lifecycle_cost'](fit_seconds,predict_seconds,100),
                    intervention=dict(column=column,feature=audit['feature_names'][column],train_correlation=float(correlation[column]),replacement=float(np.median(x['train'][:,column])),predictions=damaged.tolist(),error=s['binary_loss'](y['test'],damaged)))
                row['intervention']['delta_loss']=row['intervention']['error']-row['error'];result['records'].append(row)
                if output:Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                print(name,seed,arm,round(row['error'],5),'erase gap',round(row['intervention']['delta_loss'],5),flush=True)
    historical=json.loads((root/'_verify_l070_results.json').read_text())['records']
    old=[r for r in historical if r['dataset'] in cfg['datasets'] and r['seed'] in cfg['seeds']]
    corrected=[r for r in result['records'] if r['arm']=='TabM-mini-v2']
    main=[r for r in old if r['arm']!='TabM-mini']+corrected
    result['summary']=s['complete_panel'](main,cfg['datasets'],ARMS,cfg['seeds'])
    result['historical_summary']=s['complete_panel'](old,cfg['datasets'],ARCHIVE_ARMS,cfg['seeds'])
    result['legacy_diagnostic']=s['complete_panel'](old+corrected,cfg['datasets'],ARCHIVE_ARMS+['TabM-mini-v2'],cfg['seeds'])
    result['paired_effects']={}
    v=np.asarray(result['summary']['values'])
    for i,arm in enumerate(ARMS[1:],1):result['paired_effects'][arm]=s['dataset_bootstrap'](v[:,i,:]-v[:,0,:],2000,70)
    if identity['sha256']!=s['kernel_identity'](s,root)['sha256']:raise RuntimeError('Live definitions changed during measurement')
    result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-start)
    if output:Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

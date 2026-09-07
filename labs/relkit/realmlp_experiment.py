"""Visible L053 protocol: fixed numeric TD-S recipe versus declared XGB search.

Data are the cached TabR release, not the RealMLP benchmark splits. Results are
local protocol evidence and must not be labeled a RealMLP paper reproduction.
"""
import copy
import hashlib
import importlib.metadata
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import rankdata, friedmanchisquare, studentized_range
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor
from relkit.tabr_experiment import seed_interval
from relkit.realmlp import RobustSmooth, coslog4, last_best


SEARCH = [dict(max_depth=d, learning_rate=lr) for d in (3,6) for lr in (.03,.1,.2)]


def load_task(name, train_cap=1200, eval_cap=600, data_root='data/cache/l052'):
    arrays, labels, selection, hashes = {}, {}, {}, {}
    for j, split in enumerate(('train','val','test')):
        root=Path(data_root)/name
        for prefix in ('X_num','Y'):
            p=root/f'{prefix}_{split}.npy';hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        x=np.load(root/f'X_num_{split}.npy');y=np.load(root/f'Y_{split}.npy')
        cap=train_cap if split=='train' else eval_cap
        idx=np.random.default_rng(53+j).permutation(len(x))[:min(cap or len(x),len(x))]
        arrays[split],labels[split],selection[split]=x[idx],y[idx],idx.tolist()
    median=np.nanmedian(arrays['train'],axis=0)
    arrays={s:np.where(np.isnan(x),median,x).astype('float32') for s,x in arrays.items()}
    if not all(np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite input')
    regression=name!='higgs-small'
    mean=float(labels['train'].mean()) if regression else 0.
    std=float(labels['train'].std()) if regression else 1.
    if std<=0:raise ValueError('Constant target')
    return dict(raw=arrays,y={s:((v-mean)/std).astype('float32') for s,v in labels.items()},
                regression=regression,target_std=std,target_mean=mean,selection=selection,hashes=hashes)


def error(prediction, target, regression, target_std=1.):
    if regression:return float(np.sqrt(np.mean((np.asarray(prediction).reshape(-1)-target)**2))*target_std)
    return float(np.mean(np.asarray(prediction).argmax(-1)!=target))


def fit_neural(model_class, data, seed, width=64, epochs=64, device='cpu', prep_class=RobustSmooth):
    start=time.perf_counter();torch.manual_seed(seed)
    prep=prep_class().fit(data['raw']['train'])
    xs={s:torch.as_tensor(prep.transform(x),device=device) for s,x in data['raw'].items()}
    regression=data['regression']
    ys={s:torch.as_tensor(y,dtype=torch.float32 if regression else torch.long,device=device)
        for s,y in data['y'].items()}
    model=model_class(xs['train'].shape[1],width=width,regression=regression).to(device)
    # Weight decay is zero in TD-S; Adam and AdamW then implement the same updates.
    optimizer=torch.optim.Adam(model.parameter_groups(),betas=(.9,.95),eps=1e-8,weight_decay=0.)
    loader=DataLoader(TensorDataset(xs['train'],ys['train']),batch_size=min(256,len(xs['train'])),
                      shuffle=True,drop_last=True)
    base_lr=.07 if regression else .04
    history=[];best=float('inf');best_state=None
    for epoch in range(epochs):
        model.train()
        for batch,(x,y) in enumerate(loader):
            t=(epoch*len(loader)+batch)/(epochs*len(loader))
            for group in optimizer.param_groups:group['lr']=base_lr*group['factor']*coslog4(t)
            prediction=model(x)
            loss=nn.functional.mse_loss(prediction[:,0],y) if regression else nn.functional.cross_entropy(prediction,y,label_smoothing=.1)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            p=torch.cat([model(batch) for batch in xs['val'].split(1024)]).cpu().numpy()
        e=error(p,data['y']['val'],regression,data['target_std']);history.append(e)
        if e<=best:best=e;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    with torch.no_grad():p=torch.cat([model(batch) for batch in xs['test'].split(1024)]).cpu().numpy()
    return dict(seed=seed,error=error(p,data['y']['test'],regression,data['target_std']),
                prediction=p.tolist(),history=history,best_epoch=last_best(history)+1,
                seconds=time.perf_counter()-start)


def fit_trees(data, seed, trees=150, candidates=6):
    """One fixed first candidate, then validation-only search; test evaluated after selection."""
    start=time.perf_counter();reg=data['regression'];models=[];records=[]
    cls=XGBRegressor if reg else XGBClassifier
    for config in SEARCH[:candidates]:
        model=cls(**config,n_estimators=trees,early_stopping_rounds=20,
                  eval_metric='rmse' if reg else 'error',tree_method='hist',n_jobs=1,
                  subsample=.8,colsample_bytree=.8,random_state=seed)
        model.fit(data['raw']['train'],data['y']['train'],
                  eval_set=[(data['raw']['val'],data['y']['val'])],verbose=False)
        p=model.predict(data['raw']['val']) if reg else model.predict_proba(data['raw']['val'])
        e=error(p,data['y']['val'],reg,data['target_std'])
        models.append(model);records.append(dict(config=config,validation_error=e,best_iteration=int(model.best_iteration)))
    chosen=int(np.argmin([r['validation_error'] for r in records]))
    result={}
    for arm,index in [('XGB-fixed',0),('XGB-tuned',chosen)]:
        model=models[index]
        p=model.predict(data['raw']['test']) if reg else model.predict_proba(data['raw']['test'])
        result[arm]=dict(seed=seed,error=error(p,data['y']['test'],reg,data['target_std']),prediction=p.tolist(),
                        selected=index,search=records,seconds=time.perf_counter()-start)
    return result


def run_suite(model_class, names=('california','house','higgs-small'), seeds=(0,1,2),
              train_cap=1200,eval_cap=600,width=64,epochs=64,trees=150,candidates=6,
              device='cpu',data_root='data/cache/l052',prep_class=RobustSmooth,neural_only=False):
    start=time.perf_counter();torch.set_num_threads(1)
    arms=['RealMLP-S'] if neural_only else ['RealMLP-S','XGB-fixed','XGB-tuned']
    result=dict(config=dict(train_cap=train_cap,eval_cap=eval_cap,width=width,epochs=epochs,trees=trees,
                           candidates=candidates,seeds=list(seeds),selection_seeds=[53,54,55],device=device),
                versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','xgboost']},
                results={},verdict='INCOMPARABLE')
    with threadpool_limits(limits=1):
        for name in names:
            data=load_task(name,train_cap,eval_cap,data_root);runs={a:[] for a in arms}
            for seed in seeds:
                runs['RealMLP-S'].append(fit_neural(model_class,data,seed,width,epochs,device,prep_class))
                if not neural_only:
                    tree_runs=fit_trees(data,seed,trees,candidates)
                    for a in tree_runs:runs[a].append(tree_runs[a])
                print(name,seed,{a:round(runs[a][-1]['error'],5) for a in arms},flush=True)
            result['results'][name]=dict(metric='RMSE' if data['regression'] else 'classification error',
                runs=runs,summary={a:seed_interval([r['error'] for r in runs[a]]) for a in arms},
                selection=data['selection'],hashes=data['hashes'],target_std=data['target_std'])
    if len(names)>=3 and len(arms)>=3:
        means=np.array([[result['results'][name]['summary'][a]['mean'] for a in arms] for name in names])
        ranks=np.array([rankdata(row) for row in means]);test=friedmanchisquare(*means.T)
        cd=float(studentized_range.ppf(.95,len(arms),np.inf)/np.sqrt(2)*np.sqrt(len(arms)*(len(arms)+1)/(6*len(names))))
        result['ranks']=dict(arms=arms,per_dataset=ranks.tolist(),means=dict(zip(arms,ranks.mean(0).tolist())),
                             friedman_stat=float(test.statistic),friedman_p=float(test.pvalue),nemenyi_cd=cd)
    result['elapsed_seconds']=time.perf_counter()-start
    return result

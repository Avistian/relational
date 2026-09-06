"""Visible L052 training protocol; the supplied model class is the live notebook class."""
import copy
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata, friedmanchisquare, studentized_range, t
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from threadpoolctl import threadpool_limits


def load_task(name, train_cap=1200, eval_cap=600, data_root=None):
    """Authors' splits retained; label-blind subsamples; train-only transforms."""
    root = Path(data_root or 'data/cache/l052') / name
    arrays, labels, selection = {}, {}, {}
    for j, split in enumerate(('train', 'val', 'test')):
        x = np.load(root/f'X_num_{split}.npy'); y = np.load(root/f'Y_{split}.npy')
        cap = train_cap if split == 'train' else eval_cap
        ids = np.random.default_rng(52+j).permutation(len(x))[:min(cap or len(x),len(x))]
        arrays[split], labels[split], selection[split] = x[ids], y[ids], ids.tolist()
    median = np.nanmedian(arrays['train'],axis=0)
    for s in arrays: arrays[s] = np.where(np.isnan(arrays[s]),median,arrays[s])
    transform = QuantileTransformer(n_quantiles=min(1000,len(arrays['train'])),
                                    output_distribution='normal',random_state=52)
    transform.fit(arrays['train'])
    arrays = {s:transform.transform(x).astype('float32') for s,x in arrays.items()}
    regression = name != 'higgs-small'
    mean = float(labels['train'].mean()) if regression else 0.
    std = float(labels['train'].std()) if regression else 1.
    if std == 0: raise ValueError('Constant training target')
    labels = {s:((y-mean)/std).astype('float32') for s,y in labels.items()}
    return dict(x=arrays,y=labels,regression=regression,target_mean=mean,target_std=std,selection=selection)


def score_predictions(prediction, target, regression, target_std):
    return float(np.sqrt(mean_squared_error(target,prediction))*target_std if regression
                 else accuracy_score(target, prediction>0))


def seed_interval(values):
    values = np.asarray(values,dtype=float)
    mean = float(values.mean()); sd = float(values.std(ddof=1)) if len(values)>1 else 0.
    half = float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))) if len(values)>1 else None
    return dict(mean=mean,sd=sd,ci95=None if half is None else [mean-half,mean+half],n=len(values))


def fit_neural(model_class, data, seed, *, d=32, m=16, epochs=25, lr=.001,
               dropout=.1, context_dropout=.1, retrieval=True, device='cpu', patience=8):
    torch.manual_seed(seed)
    model = model_class(data['x']['train'].shape[1],d=d,m=m,dropout=dropout,
                        context_dropout=context_dropout,regression=data['regression'],retrieval=retrieval).to(device)
    xs = {s:torch.tensor(x,device=device) for s,x in data['x'].items()}
    ys = {s:torch.tensor(y,device=device) for s,y in data['y'].items()}
    ids = torch.arange(len(xs['train']),device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    best, best_state, history, stale = float('inf'), None, [], 0
    def predict(split, candidate_y=None):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(batch,xs['train'],ys['train'] if candidate_y is None else candidate_y)
                              for batch in xs[split].split(256)]).cpu().numpy()
    for epoch in range(epochs):
        model.train()
        for batch_ids in torch.randperm(len(ids),device=device).split(256):
            logits = model(xs['train'][batch_ids],xs['train'],ys['train'],batch_ids,ids)
            loss = (nn.functional.mse_loss(logits,ys['train'][batch_ids]) if data['regression']
                    else nn.functional.binary_cross_entropy_with_logits(logits,ys['train'][batch_ids]))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
        prediction = predict('val')
        score = score_predictions(prediction,data['y']['val'],data['regression'],data['target_std'])
        objective = score if data['regression'] else -score
        history.append(score)
        if objective < best:
            best, best_state, stale = objective, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if stale >= patience: break
    model.load_state_dict(best_state)
    test_prediction = predict('test')
    result = dict(seed=seed,score=score_predictions(test_prediction,data['y']['test'],data['regression'],data['target_std']),
                  prediction=test_prediction.tolist(),validation_history=history,best_epoch=int(np.argmin(history) if data['regression'] else np.argmax(history))+1)
    if retrieval:
        # Inference intervention, not a retrained ablation: same keys, different stored labels.
        shuffled = ys['train'][torch.as_tensor(np.random.default_rng(52).permutation(len(ids)),device=device)]
        altered = predict('test',shuffled)
        result['shuffled_label_score'] = score_predictions(altered,data['y']['test'],data['regression'],data['target_std'])
    return result


def run_suite(model_class, *, names=('california','house','higgs-small'), seeds=(0,1,2),
              train_cap=1200, eval_cap=600, d=32, m=16, epochs=25, trees=150,
              lr=.001, dropout=.1, context_dropout=.1, patience=8, device='cpu',
              data_root='data/cache/l052', arms=('MLP','TabR-S','XGBoost')):
    torch.set_num_threads(1)
    start = time.time(); results = {}
    with threadpool_limits(limits=1):
        for name in names:
            data = load_task(name,train_cap,eval_cap,data_root)
            runs = {a:[] for a in arms}
            for seed in seeds:
                for arm in arms:
                    if arm != 'XGBoost':
                        run = fit_neural(model_class,data,seed,d=d,m=m,epochs=epochs,lr=lr,
                                         dropout=dropout,context_dropout=context_dropout,
                                         patience=patience,retrieval=arm=='TabR-S',device=device)
                    else:
                        cls = XGBRegressor if data['regression'] else XGBClassifier
                        tree = cls(n_estimators=trees,max_depth=4,learning_rate=.05,subsample=.8,
                                   colsample_bytree=.8,n_jobs=1,random_state=seed,tree_method='hist',early_stopping_rounds=16)
                        tree.fit(data['x']['train'],data['y']['train'],eval_set=[(data['x']['val'],data['y']['val'])],verbose=False)
                        prediction = tree.predict(data['x']['test']) if data['regression'] else tree.predict_proba(data['x']['test'])[:,1]-.5
                        run = dict(seed=seed,score=score_predictions(prediction,data['y']['test'],data['regression'],data['target_std']),prediction=prediction.tolist(),best_epoch=int(tree.best_iteration)+1)
                    runs[arm].append(run)
                    print(name,seed,arm,round(run['score'],5),flush=True)
            results[name] = dict(metric='RMSE' if data['regression'] else 'accuracy',
                target_std=data['target_std'],test_target=data['y']['test'].tolist(),selection=data['selection'],
                runs=runs,summary={a:seed_interval([r['score'] for r in rr]) for a,rr in runs.items()})
    stats = None
    if len(names)>=3 and len(arms)>=3:
        scores = np.array([[r['summary'][a]['mean']*(1 if r['metric']=='RMSE' else -1) for a in arms] for r in results.values()])
        ranks = np.array([rankdata(row) for row in scores]); k=len(arms);n=len(names)
        stats = dict(arms=list(arms),mean_ranks=ranks.mean(0).tolist(),dataset_ranks=ranks.tolist(),
                     friedman_p=float(friedmanchisquare(*ranks.T).pvalue),
                     cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n))))
    return dict(results=results,stats=stats,elapsed_seconds=time.time()-start,
                versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','xgboost']},
                hardware=platform.machine()+' '+device,verdict='INCOMPARABLE',
                protocol=dict(train_cap=train_cap,eval_cap=eval_cap,seeds=list(seeds),subsample_seed=52,
                    split='author-provided; no resplitting',d=d,m=m,epochs=epochs,trees=trees,lr=lr,
                    dropout=dropout,context_dropout=context_dropout,patience=patience,
                    preprocessing='train median and normal quantiles; train target standardization; no quantile jitter',arms=list(arms)))

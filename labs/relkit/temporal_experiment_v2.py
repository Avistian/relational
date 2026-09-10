"""TabReD split intervention using existing numeric MLP/TabM-mini and XGBoost.

This small experiment is INCOMPARABLE to Figure 2. It uses the 2026 release,
three tasks, two compact candidates per arm, one paired split and three seeds.
Version 2 uses corrected tabm_v2 and shuffles exactly the capped temporal pool.
Historical temporal_experiment.py and its results are intentionally unchanged.
"""
import copy
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata, friedmanchisquare, studentized_range
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
from threadpoolctl import threadpool_limits
from relkit.tabm_v2 import MLP, TabM
from relkit.temporal_v2 import fit_preprocessor, apply_preprocessor, select_candidate, rank_change, paired_random_split

TASKS = ('ecom-offers', 'homesite-insurance', 'sberbank-housing')
ARMS = ('MLP', 'TabM-mini', 'XGBoost')


def load_release(name, strategy, split_id=0, train_cap=1500, eval_cap=600, root='data/cache/l055'):
    folder = Path(root) / name / name
    reg = json.loads((folder / 'info.json').read_text())['task']['type'] == 'regression'
    matrices = [np.load(folder / (k + '.npy'), mmap_mode='r') for k in ('x_num', 'x_bin') if (folder / (k + '.npy')).exists()]
    y = np.load(folder / 'y.npy')
    timestamps = np.load(folder / 'x_meta.npy', mmap_mode='r')[:, 0]
    if strategy not in ('random', 'temporal'): raise ValueError('Unknown protocol')
    split_name = 'sliding-window-' + str(split_id)
    all_ids = {s: np.load(folder / 'splits' / split_name / (s + '.npy')) for s in ('train', 'val', 'test')}
    assert sum(map(len, all_ids.values())) == len(np.unique(np.concatenate(list(all_ids.values()))))
    ids = {}
    for i, s in enumerate(all_ids):
        cap = train_cap if s == 'train' else eval_cap
        pool = all_ids[s]
        ids[s] = np.sort(np.random.default_rng(550 + i).choice(pool, min(cap, len(pool)), replace=False))
    pool_hash = hashlib.sha256(np.sort(np.concatenate(list(ids.values()))).tobytes()).hexdigest()
    if strategy == 'random':
        ids = paired_random_split(ids, 5550 + split_id)
    x = {s: np.concatenate([m[idx] for m in matrices], axis=1) for s, idx in ids.items()}
    audit = dict(split=split_name, assignment=('capped-pool-shuffle' if strategy == 'random' else 'released-temporal-subset'),
                 pool_hash=pool_hash, shuffle_seed=5550+split_id, full_sizes={s: len(v) for s, v in all_ids.items()},
                 sizes={s: len(v) for s, v in ids.items()}, features=x['train'].shape[1],
                 omitted_categorical_features=(np.load(folder/'x_cat.npy', mmap_mode='r').shape[1] if (folder/'x_cat.npy').exists() else 0),
                 time_ranges={s: [int(timestamps[v].min()), int(timestamps[v].max())] for s, v in ids.items()},
                 index_hashes={s: hashlib.sha256(v.tobytes()).hexdigest() for s, v in ids.items()})
    if strategy == 'temporal':
        for a, b in [('train','val'), ('val','test')]:
            assert timestamps[all_ids[a]].max() <= timestamps[all_ids[b]].min(), 'Released temporal split is not ordered'
        audit['strict_time_boundaries'] = all(timestamps[all_ids[a]].max() < timestamps[all_ids[b]].min() for a,b in [('train','val'),('val','test')])
    return x, {s: y[v] for s,v in ids.items()}, reg, audit


def metric(y, prediction, regression):
    return float(np.sqrt(np.mean((y-prediction)**2))) if regression else float(1-roc_auc_score(y, prediction))


def fit_neural(x, y, reg, arm, seed, width=64, epochs=32, lr=.001, k=8):
    """Training and checkpoint selection receive train/validation only."""
    torch.manual_seed(seed)
    xs = {s: torch.tensor(v) for s,v in x.items()}
    mu, sd = (float(y['train'].mean()), max(float(y['train'].std()), 1e-8)) if reg else (0., 1.)
    target = torch.tensor((y['train']-mu)/sd, dtype=torch.float32 if reg else torch.long)
    cls = TabM if arm == 'TabM-mini' else MLP
    kwargs = dict(din=x['train'].shape[1], width=width, depth=2, dropout=.1, regression=reg, k=k)
    if arm == 'TabM-mini': kwargs.update(arch='mini', seed=seed)
    model = cls(**kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    def predict(values):
        model.eval()
        with torch.no_grad():
            out = model.predict(values) if arm == 'TabM-mini' else model(values)
            if arm == 'MLP': out = out[:,0] if reg else out.softmax(-1)
            out = out.numpy()
        return out*sd+mu if reg else out[:,1]
    best, state, best_epoch = float('inf'), None, None
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(target)).split(256):
            if arm == 'TabM-mini': loss = model.member_losses(xs['train'][idx], target[idx])
            else:
                out = model(xs['train'][idx])
                loss = nn.functional.mse_loss(out[:,0], target[idx]) if reg else nn.functional.cross_entropy(out, target[idx])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(); loss.backward(); opt.step()
        val = metric(y['val'], predict(xs['val']), reg)
        if val < best:
            best, state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch+1
    model.load_state_dict(state)
    return lambda values: predict(torch.tensor(values)), best, best_epoch


def evaluate_arm(x, y, reg, arm, seed, epochs=32, trees=120):
    """Two candidates; validation selects before test predictions are computed."""
    candidates = []
    for candidate in range(2):
        if arm == 'XGBoost':
            cls = XGBRegressor if reg else XGBClassifier
            model = cls(n_estimators=trees, max_depth=(3,6)[candidate], learning_rate=.05,
                        subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=seed, tree_method='hist')
            model.fit(x['train'], y['train'])
            predict = model.predict if reg else lambda values, m=model: m.predict_proba(values)[:,1]
            val, epoch = metric(y['val'], predict(x['val']), reg), None
        else:
            predict, val, epoch = fit_neural({s:x[s] for s in ('train','val')}, {s:y[s] for s in ('train','val')},
                                           reg, arm, seed, epochs=epochs, lr=(.001,.003)[candidate])
        candidates.append((predict, val, epoch))
    selected = select_candidate([c[1] for c in candidates])
    predict, val, epoch = candidates[selected]
    pred = predict(x['test'])
    return dict(seed=seed, selected=selected, validation_errors=[c[1] for c in candidates], best_epoch=epoch,
                error=metric(y['test'], pred, reg), predictions=pred.tolist())


def run_suite(names=TASKS, seeds=(0,1,2), split_ids=(0,), train_cap=1500, eval_cap=600,
              epochs=32, trees=120, root='data/cache/l055'):
    torch.set_num_threads(1)
    start = time.perf_counter()
    result = dict(operator='temporal_experiment_v2', verdict='INCOMPARABLE', config=dict(names=list(names), seeds=list(seeds), split_ids=list(split_ids),
                  train_cap=train_cap, eval_cap=eval_cap, epochs=epochs, trees=trees, row_sampling_seeds=[550,551,552]),
                  versions={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','scikit-learn','xgboost')},
                  tasks={})
    with threadpool_limits(limits=1):
        for name in names:
            result['tasks'][name] = {}
            for strategy in ('random','temporal'):
                records = []
                for split_id in split_ids:
                    raw, y, reg, audit = load_release(name,strategy,split_id,train_cap,eval_cap,root)
                    state = fit_preprocessor(raw['train'])
                    x = {s:apply_preprocessor(v,state) for s,v in raw.items()}
                    record = dict(audit=audit, metric='RMSE of released log target' if reg else '1 - AUROC',
                                  test_targets=y['test'].tolist(), arms={})
                    for arm in ARMS:
                        record['arms'][arm] = [evaluate_arm(x,y,reg,arm,seed,epochs,trees) for seed in seeds]
                        print(name,strategy,split_id,arm,round(np.mean([v['error'] for v in record['arms'][arm]]),5),flush=True)
                    records.append(record)
                result['tasks'][name][strategy] = records
    result['seconds'] = time.perf_counter()-start
    return result


def summarize(result):
    """Average seeds then split windows within each task; datasets are rank units."""
    summary = {}
    for strategy in ('random','temporal'):
        errors = np.array([[np.mean([v['error'] for rec in task[strategy] for v in rec['arms'][arm]])
                            for arm in ARMS] for task in result['tasks'].values()])
        ranks = rankdata(errors,axis=1)
        n,k = errors.shape
        summary[strategy] = dict(errors=errors.tolist(), ranks=ranks.tolist(), mean_ranks=ranks.mean(0).tolist())
        if n >= 3:
            summary[strategy].update(friedman_p=float(friedmanchisquare(*errors.T).pvalue),
                nemenyi_cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n))))
    return summary

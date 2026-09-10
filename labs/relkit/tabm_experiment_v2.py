"""L054 protocol: does the gain come from a better model, a recipe, or an ensemble?

We compare, on the reused L052 numeric archive (same caps as L053), four arms:

    MLP        one plain MLP                       (the k=1 individual model)
    MLP-xk    deep ensemble of 32 MLPs, averaged  (the expensive ensemble)
    TabM-mini  one TabM (k=32), averaged           (the parameter-efficient ensemble)
    XGB-tuned  6-candidate XGBoost search          (continuity baseline, from L053)

We also record TabM's mean *individual* submodel error next to its *collective*
error, to probe the paper's headline (§5.1): submodels are weak individually
but strong collectively. Data are the cached TabR release, not the TabM benchmark
splits; the paper verdict is INCOMPARABLE.
"""
import copy
import importlib.metadata
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import rankdata, friedmanchisquare, studentized_range
from threadpoolctl import threadpool_limits
from relkit.tabr_experiment import seed_interval
from relkit.realmlp_experiment import load_task, error, fit_trees
from relkit.tabm_v2 import MLP, TabM


def _standardise(x):
    """Fit z-score on training rows only (plain preprocessing for MLP/TabM)."""
    mean = x['train'].mean(0, keepdims=True)
    std = x['train'].std(0, keepdims=True)
    std = np.where(std > 0, std, 1.)
    return {s: ((v - mean) / std).astype('float32') for s, v in x.items()}


def _fit_one_mlp(data, xs, ys, seed, width, depth, epochs, lr, dropout, device='cpu'):
    """Train one plain MLP; return its best-epoch state and validation error."""
    torch.manual_seed(seed)
    regression = data['regression']
    model = MLP(xs['train'].shape[1], width=width, depth=depth,
                dropout=dropout, regression=regression).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    loader = DataLoader(TensorDataset(xs['train'], ys['train']),
                        batch_size=min(256, len(xs['train'])), shuffle=True, drop_last=True)
    best, best_state, history = float('inf'), None, []
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            out = model(x)
            loss = nn.functional.mse_loss(out[:, 0], y) if regression \
                else nn.functional.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(xs['val'])
            p = p[:, 0].cpu().numpy() if regression else p.softmax(-1).cpu().numpy()
        e = error(p, data['y']['val'], regression, data['target_std'])
        history.append(e)
        if e < best:
            best, best_state = e, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def _test_pred(model, xs, regression):
    model.eval()
    with torch.no_grad():
        p = model(xs['test'])
        return p[:, 0].cpu().numpy() if regression else p.softmax(-1).cpu().numpy()


def fit_mlp(data, xs, ys, seed, **hp):
    start = time.perf_counter()
    model, history = _fit_one_mlp(data, xs, ys, seed, **hp)
    p = _test_pred(model, xs, data['regression'])
    return dict(seed=seed, error=error(p, data['y']['test'], data['regression'], data['target_std']),
                prediction=p.tolist(), history=history, best_epoch=int(np.argmin(history)) + 1, parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter() - start)


def fit_deep_ensemble(data, xs, ys, seed, members=32, **hp):
    """MLP^{×k}: train `members` independent MLPs, average their test predictions."""
    start = time.perf_counter(); reg = data['regression']; preds = []
    for i in range(members):
        model, _ = _fit_one_mlp(data, xs, ys, seed * 1000 + i, **hp)
        preds.append(_test_pred(model, xs, reg))
    mean = np.mean(preds, axis=0)
    return dict(seed=seed, error=error(mean, data['y']['test'], reg, data['target_std']),
                prediction=mean.tolist(), members=members, seconds=time.perf_counter() - start)


def fit_tabm(data, xs, ys, seed, k=32, width=64, depth=3, epochs=64, lr=2e-3,
             dropout=.1, arch='mini', device='cpu'):
    """One TabM; record collective and mean-individual test errors (diversity)."""
    start = time.perf_counter(); torch.manual_seed(seed); reg = data['regression']
    model = TabM(xs['train'].shape[1], k=k, width=width, depth=depth, dropout=dropout,
                 regression=reg, arch=arch, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    loader = DataLoader(TensorDataset(xs['train'], ys['train']),
                        batch_size=min(256, len(xs['train'])), shuffle=True, drop_last=True)
    best, best_state, history = float('inf'), None, []
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            loss = model.member_losses(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model.predict(xs['val']).cpu().numpy()
        e = error(p, data['y']['val'], reg, data['target_std']); history.append(e)
        if e < best:
            best, best_state = e, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        collective = model.predict(xs['test']).cpu().numpy()
        members = model.member_predictions(xs['test']).cpu().numpy()   # [B,k] or [B,k,C]
    coll_err = error(collective, data['y']['test'], reg, data['target_std'])
    ind_errs = [error(members[:, i], data['y']['test'], reg, data['target_std'])
                for i in range(k)]
    return dict(seed=seed, error=coll_err, best_epoch=int(np.argmin(history)) + 1,
                individual_mean=float(np.mean(ind_errs)), individual_sd=float(np.std(ind_errs)),
                individual_best=float(np.min(ind_errs)), prediction=collective.tolist(), member_predictions=members.tolist(), history=history, parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter() - start)


def run_suite(names=('california', 'house', 'higgs-small'), seeds=(0, 1, 2),
              k=32, width=64, depth=3, epochs=64, lr=2e-3, dropout=.1,
              train_cap=1200, eval_cap=600, trees=150, candidates=6,
              device='cpu', data_root='data/cache/l052'):
    start = time.perf_counter(); torch.set_num_threads(1)
    arms = ['MLP', 'MLP-xk', 'TabM-mini', 'XGB-tuned']
    hp = dict(width=width, depth=depth, epochs=epochs, lr=lr, dropout=dropout)
    result = dict(config=dict(k=k, train_cap=train_cap, eval_cap=eval_cap, **hp,
                              trees=trees, candidates=candidates, seeds=list(seeds),
                              selection_seeds=[53, 54, 55], device=device),
                  versions={p: importlib.metadata.version(p)
                            for p in ['torch', 'numpy', 'scipy', 'scikit-learn', 'xgboost']},
                  results={}, verdict='INCOMPARABLE')
    with threadpool_limits(limits=1):
        for name in names:
            data = load_task(name, train_cap, eval_cap, data_root)
            xr = _standardise(data['raw'])
            xs = {s: torch.as_tensor(v, device=device) for s, v in xr.items()}
            ys = {s: torch.as_tensor(data['y'][s],
                                     dtype=torch.float32 if data['regression'] else torch.long,
                                     device=device) for s in xr}
            runs = {a: [] for a in arms}
            for seed in seeds:
                runs['MLP'].append(fit_mlp(data, xs, ys, seed, device=device, **hp))
                runs['MLP-xk'].append(fit_deep_ensemble(data, xs, ys, seed, members=k, device=device, **hp))
                runs['TabM-mini'].append(fit_tabm(data, xs, ys, seed, k=k, arch='mini',
                                                  width=width, depth=depth, epochs=epochs,
                                                  lr=lr, dropout=dropout, device=device))
                runs['XGB-tuned'].append(fit_trees(data, seed, trees, candidates)['XGB-tuned'])
                print(name, seed, {a: round(runs[a][-1]['error'], 5) for a in arms}, flush=True)
            result['results'][name] = dict(
                metric='RMSE' if data['regression'] else 'classification error',
                runs=runs, summary={a: seed_interval([r['error'] for r in runs[a]]) for a in arms},
                diversity=seed_interval([r['individual_mean'] for r in runs['TabM-mini']]),
                selection=data['selection'], hashes=data['hashes'], target_std=data['target_std'])
    means = np.array([[result['results'][n]['summary'][a]['mean'] for a in arms] for n in names])
    ranks = np.array([rankdata(row) for row in means]); test = friedmanchisquare(*means.T)
    cd = float(studentized_range.ppf(.95, len(arms), np.inf) / np.sqrt(2)
               * np.sqrt(len(arms) * (len(arms) + 1) / (6 * len(names))))
    result['ranks'] = dict(arms=arms, per_dataset=ranks.tolist(),
                           means=dict(zip(arms, ranks.mean(0).tolist())),
                           friedman_stat=float(test.statistic), friedman_p=float(test.pvalue),
                           nemenyi_cd=cd)
    result['elapsed_seconds'] = time.perf_counter() - start
    return result

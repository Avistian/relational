"""L050: numeric FT-Transformer and a validation-selected comparison.

Paper: Gorishniy et al. 2021, §3.3 / Appendix E. Numeric-only, no compression,
ReGLU, skip first attention normalization. All-token final attention is retained;
only CLS is read, equivalent to the reference's final CLS-only query in eval.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
import time
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import t, rankdata, friedmanchisquare, studentized_range, binomtest
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from relkit.data import load_tier_a, CACHE, SPECS


def prepare_numeric(x, train):
    """Fit median imputation and population mean/SD on training rows only."""
    x = np.asarray(x, dtype=float)
    median = np.nanmedian(x[train], axis=0)
    if not np.isfinite(median).all():
        raise ValueError('Entirely missing training feature: choose a policy explicitly.')
    filled = np.where(np.isnan(x), median, x)
    mean, scale = filled[train].mean(0), filled[train].std(0)
    scale = np.where(scale == 0, 1, scale)
    return ((filled - mean) / scale).astype('float32'), {
        'median': median.tolist(), 'mean': mean.tolist(), 'scale': scale.tolist()}


def select_trial(validation_scores):
    """Higher AUROC wins; exact ties go to the first declared candidate."""
    scores = np.asarray(validation_scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Need finite validation scores.')
    return int(np.argmax(scores))


def paired_summary(a, b):
    """Conditional training-seed interval; not a population or split interval."""
    delta = np.asarray(a) - np.asarray(b)
    if len(delta) < 2:
        return {'mean': float(delta.mean()), 'sd': None, 'ci95': None}
    mean, sd = float(delta.mean()), float(delta.std(ddof=1))
    half = float(t.ppf(.975, len(delta)-1) * sd / np.sqrt(len(delta)))
    return {'mean': mean, 'sd': sd, 'ci95': [mean-half, mean+half]}


def reglu(x):
    """Appendix E: split the 2h coordinates, then a * max(b, 0)."""
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class CheckpointAttention(nn.Module):
    """§3.3: independent Q/K/V maps, scaled scores, weighted values, projection."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = [nn.Linear(d, d) for _ in range(4)]
        self.drop = nn.Dropout(dropout)
        for layer in (self.q, self.k, self.v, self.out):
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        b, c, d = x.shape
        def split(layer):
            return layer(x).reshape(b, c, self.heads, d//self.heads).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        weights = torch.softmax(q @ k.transpose(-1, -2) / (d//self.heads)**.5, -1)
        mixed = (self.drop(weights) @ v).transpose(1, 2).reshape(b, c, d)
        return self.out(mixed)


class CheckpointBlock(nn.Module):
    """First attention has no LayerNorm; FFN always has pre-normalization."""
    def __init__(self, d, heads, first, attention_dropout, ffn_dropout):
        super().__init__()
        self.norm1 = nn.Identity() if first else nn.LayerNorm(d)
        self.attention = CheckpointAttention(d, heads, attention_dropout)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * 4/3)
        self.linear1, self.linear2 = nn.Linear(d, 2*hidden), nn.Linear(hidden, d)
        self.drop = nn.Dropout(ffn_dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.linear2(self.drop(reglu(self.linear1(self.norm2(x)))))


class CheckpointFT(nn.Module):
    """Full numeric prediction path; residual dropout fixed at zero."""
    def __init__(self, features, d=32, layers=2, heads=4, attention_dropout=.1, ffn_dropout=.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, d))
        self.bias = nn.Parameter(torch.empty(features, d))
        self.cls = nn.Parameter(torch.empty(d))
        for p in (self.weight, self.bias, self.cls):
            nn.init.uniform_(p, -d**-.5, d**-.5)
        self.blocks = nn.ModuleList([CheckpointBlock(d, heads, i == 0, attention_dropout, ffn_dropout) for i in range(layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), 1, -1), tokens], 1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(F.relu(self.norm(tokens[:, 0]))).squeeze(-1)


class CheckpointMLP(nn.Module):
    """§3.1 baseline: two linear/ReLU/dropout layers and one binary head."""
    def __init__(self, features, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features,d), nn.ReLU(), nn.Dropout(.1),
                                 nn.Linear(d,d), nn.ReLU(), nn.Dropout(.1), nn.Linear(d,1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def reference_parity():
    """Copy reference weights; check logits and input gradients (eval, numeric path)."""
    import rtdl_revisiting_models as rtdl
    torch.manual_seed(50)
    reference = rtdl.FTTransformer(n_cont_features=4,cat_cardinalities=[],d_out=1,
        n_blocks=2,d_block=16,attention_n_heads=4,attention_dropout=.1,
        ffn_d_hidden_multiplier=4/3,ffn_dropout=.1,residual_dropout=0.).double().eval()
    own = CheckpointFT(4,d=16,layers=2,heads=4).double().eval()
    def cp(a,b): a.data.copy_(b.data)
    cp(own.weight,reference.cont_embeddings.weight); cp(own.bias,reference.cont_embeddings.bias)
    cp(own.cls,reference.cls_embedding.weight)
    for i,(a,b) in enumerate(zip(own.blocks,reference.backbone.blocks)):
        for local, remote in [('q','W_q'),('k','W_k'),('v','W_v'),('out','W_out')]:
            getattr(a.attention,local).load_state_dict(getattr(b['attention'],remote).state_dict())
        if i: a.norm1.load_state_dict(b['attention_normalization'].state_dict())
        a.norm2.load_state_dict(b['ffn_normalization'].state_dict())
        a.linear1.load_state_dict(b['ffn'].linear1.state_dict())
        a.linear2.load_state_dict(b['ffn'].linear2.state_dict())
    own.norm.load_state_dict(reference.backbone.output.normalization.state_dict())
    own.head.load_state_dict(reference.backbone.output.linear.state_dict())
    x = torch.randn(7,4,dtype=torch.double,requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    a,b = own(x), reference(xr,None).squeeze(-1)
    a.sum().backward(); b.sum().backward()
    forward, gradient = (a-b).abs().max().item(), (x.grad-xr.grad).abs().max().item()
    assert forward < 1e-10 and gradient < 1e-10, (forward,gradient)
    return {'max_logit_error':forward,'max_input_gradient_error':gradient,
            'reference_version':importlib.metadata.version('rtdl_revisiting_models'),
            'reference_sha256':hashlib.sha256(Path(inspect.getfile(rtdl)).read_bytes()).hexdigest(),
            'scope':'numeric, 2 blocks, 4 heads, no compression, eval; not optimizer/dropout parity'}


def train_neural(model, xtrain, ytrain, xvalid, yvalid, *, seed, lr, epochs, patience=8, device='cpu'):
    """Validation-only early stopping; hold out test inputs until after selection."""
    model.to(device)
    xt, xv = [torch.as_tensor(x,device=device) for x in (xtrain,xvalid)]
    yt = torch.as_tensor(ytrain,dtype=torch.float32,device=device)
    # Follow reference exclusion: no decay on embeddings, biases or normalization.
    decay, no_decay = [], []
    for name,p in model.named_parameters():
        (no_decay if p.ndim < 2 or name in ('weight','bias','cls') else decay).append(p)
    opt = torch.optim.AdamW([{'params':decay,'weight_decay':1e-5},
                            {'params':no_decay,'weight_decay':0.}],lr=lr)
    generator = torch.Generator().manual_seed(seed)
    best, state, wait, best_epoch = -1., None, 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt),generator=generator).split(256):
            idx = idx.to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xt[idx]),yt[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): score = roc_auc_score(yvalid,model(xv).sigmoid().cpu().numpy())
        history.append(float(score))
        if score > best:
            best, state, wait, best_epoch = float(score), copy.deepcopy(model.state_dict()), 0, epoch+1
        else: wait += 1
        if wait >= patience: break
    model.load_state_dict(state)
    return model, {'validation_auc':best,'best_step':best_epoch,'history':history}


def predict(model, x, device='cpu'):
    if isinstance(model, XGBClassifier): return model.predict_proba(x)[:,1]
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(chunk,device=device)).sigmoid().cpu().numpy()
                               for chunk in np.array_split(x,max(1,int(np.ceil(len(x)/1024))))])


def run_comparison(*, datasets=('diabetes','blood_transfusion','phoneme'), seeds=(0,1,2),
                   epochs=35, trees=160, d=32, layers=2, cap=None, device='cpu',
                   model_factory=CheckpointFT, prepare=prepare_numeric, selector=select_trial):
    """Two predeclared candidates per arm/seed; all use validation AUROC.

    Dataset loading/splitting is harness code; model_factory/prepare/selector keep
    the notebook's visible model and student functions on the measured path.
    """
    torch.set_num_threads(1)
    models = ['FT-Transformer','XGBoost','MLP']
    result = {'datasets':{},'models':models,'seeds':list(seeds),'split_seed':50,
              'settings':dict(epochs=epochs,trees=trees,d=d,layers=layers,cap=cap,device=device,
                              trials=2,neural_lrs=[.001,.0003],xgb_depths=[3,6],
                              xgb_lr=.05,xgb_subsample=.8,xgb_colsample=.8,patience_neural=8,patience_xgb=20),
              'scope':'local fixed split and two-candidate search; INCOMPARABLE to paper tables'}
    start = time.perf_counter()
    for name in datasets:
        frame,y = load_tier_a(name); y = np.asarray(y,dtype=int)
        assert set(np.unique(y)) == {0,1}, 'This checkpoint requires binary 0/1 labels.'
        if any(str(t) in ('category','object') for t in frame.dtypes):
            raise ValueError('Numeric-only track; categorical encoding is outside this preset.')
        raw = frame.to_numpy(dtype=float)
        source_rows = np.arange(len(y))
        if cap and len(y)>cap:
            # Label-independent subsample; exact original row IDs saved.
            source_rows = np.sort(np.random.default_rng(50).choice(len(y),cap,replace=False))
            raw,y = raw[source_rows],y[source_rows]
        tr,te = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=50)
        tr,va = train_test_split(tr,test_size=.25,stratify=y[tr],random_state=50)
        x,prep = prepare(raw,tr)
        row = {'openml_id':SPECS[name]['openml_id'], 'n':len(y),'features':frame.columns.tolist(),
               'source_sha256':hashlib.sha256((CACHE/f'{name}.parquet').read_bytes()).hexdigest(),
               'source_rows':source_rows.tolist(),'split':{'train':tr.tolist(),'valid':va.tolist(),'test':te.tolist()},
               'preprocessing':prep,'y_test':y[te].tolist(),'runs':{m:[] for m in models}}
        for seed in seeds:
            for arm in models:
                candidates,logs = [],[]
                for trial in range(2):
                    torch.manual_seed(seed)
                    tick = time.perf_counter()
                    if arm == 'XGBoost':
                        model = XGBClassifier(n_estimators=trees,max_depth=[3,6][trial],learning_rate=.05,
                            subsample=.8,colsample_bytree=.8,reg_lambda=1.,tree_method='hist',n_jobs=1,
                            eval_metric='auc',early_stopping_rounds=20,random_state=seed)
                        model.fit(x[tr],y[tr],eval_set=[(x[va],y[va])],verbose=False)
                        log = {'validation_auc':float(model.best_score),'best_step':model.best_iteration+1,
                               'history':model.evals_result()['validation_0']['auc']}
                    else:
                        model = model_factory(x.shape[1],d=d,layers=layers,heads=4) if arm == 'FT-Transformer' else CheckpointMLP(x.shape[1],d=64)
                        model,log = train_neural(model,x[tr],y[tr],x[va],y[va],seed=seed,
                            lr=[.001,.0003][trial],epochs=epochs,device=device)
                    log.update(trial=trial,elapsed_s=time.perf_counter()-tick)
                    candidates.append(model); logs.append(log)
                chosen = selector([a['validation_auc'] for a in logs])
                # The first and only test access for this arm/seed, AFTER selection.
                p = predict(candidates[chosen],x[te],device)
                record = {'seed':seed,'selected_trial':chosen,'trials':logs,'probability':p.tolist(),
                          'auc':float(roc_auc_score(y[te],p)), 'accuracy':float(accuracy_score(y[te],p>=.5))}
                row['runs'][arm].append(record)
                print(name,seed,arm,round(record['auc'],5),'candidate',chosen,flush=True)
        row['summary'] = {m:paired_summary([r['auc'] for r in row['runs'][m]],np.zeros(len(seeds))) for m in models}
        row['ft_minus_xgb'] = paired_summary([r['auc'] for r in row['runs'][models[0]]],
                                             [r['auc'] for r in row['runs'][models[1]]])
        result['datasets'][name] = row
    score = np.array([[row['summary'][m]['mean'] for m in models] for row in result['datasets'].values()])
    ranks = np.array([rankdata(-r,method='average') for r in score])
    result['mean_ranks'] = dict(zip(models,ranks.mean(0).tolist()))
    if len(score)>1:
        stat,p = friedmanchisquare(*score.T)
        wins = int(np.sum(score[:,0]>score[:,1])); losses = int(np.sum(score[:,0]<score[:,1]))
        result['statistics'] = {'friedman_chi2':float(stat) if np.isfinite(stat) else None,
            'friedman_p':float(p) if np.isfinite(p) else None,
            'nemenyi_cd':float(studentized_range.ppf(.95,3,np.inf)/np.sqrt(2)*np.sqrt(3*4/(6*len(score)))),
            'exact_sign_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
            'ft_wins':wins,'ft_losses':losses,'independent_units':len(score)}
    result['elapsed_s'] = time.perf_counter()-start
    result['versions'] = {p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','scikit-learn','xgboost')}
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result

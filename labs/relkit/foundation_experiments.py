"""Executable, bounded experiments; all outputs label measurement scope explicitly."""
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata,friedmanchisquare,studentized_range
from relkit.foundation_core import CountPFN,RowPFN,posterior_predictive,sample_scm,sample_coin_tasks,temporal_eligible
from relkit.benchmark_core import null_search,dataset_bootstrap,attention_cost

ROOT=Path(__file__).resolve().parents[1]


def parse_talent(path):
    """Parse mean+SD releases; keep missing values missing and strip emphasis only."""
    lines=[line for line in Path(path).read_text().splitlines() if line.startswith('|')]
    columns=[v.strip() for v in lines[0].strip('|').split('|')]
    records=[]
    for line in lines[2:]:
        cells=[v.strip().replace('*','') for v in line.strip('|').split('|')]
        if len(cells)!=len(columns):raise ValueError('Malformed released table')
        row={'dataset':cells[0]}
        for name,value in zip(columns[1:],cells[1:]):
            match=re.fullmatch(r'([\d.eE+-]+)\+([\d.eE+-]+)',value)
            row[name]=float(match.group(1)) if match else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index('dataset')


def survey_audit():
    arms=['xgboost','catboost','mlp','realmlp','tabr','ftt']
    tables={};ranks=[];coverage={};dataset_rows=[]
    for task,file in [('binary','cls_bin.md'),('multiclass','cls_multi.md'),('regression','regression.md')]:
        frame=parse_talent(ROOT/'sources/l058'/file)[arms]
        complete=frame.dropna();coverage[task]=dict(released=len(frame),complete=len(complete),excluded=frame.index[frame.isna().any(axis=1)].tolist())
        values=complete.to_numpy();rr=np.array([rankdata(v if task=='regression' else -v) for v in values])
        ranks.extend(rr);dataset_rows.extend(dict(dataset=n,task=task,ranks=dict(zip(arms,r.tolist()))) for n,r in zip(complete.index,rr))
        tables[task]=dict(zip(arms,rr.mean(0).tolist()))
    ranks=np.array(ranks);interval=dataset_bootstrap(ranks)
    # Deliberately outcome-selected subset. This is an illustration of selection bias.
    subset=np.argsort(ranks[:,0]-ranks[:,2])[:45]
    return dict(scope='Frozen TALENT result reanalysis; no training or seed-level data',arms=arms,coverage=coverage,
        datasets=len(ranks),per_task_ranks=tables,mean_ranks=dict(zip(arms,ranks.mean(0).tolist())),
        bootstrap_t95=interval.tolist(),xgb_favored_45_ranks=dict(zip(arms,ranks[subset].mean(0).tolist())),
        dataset_rows=dataset_rows,verdict='INCOMPARABLE',
        limitation='Current released rounded means, six-method pool, complete cases; not the full paper ranking or current corrected benchmark')


def validation_audit():
    rows=[]
    for seed in range(200):
        for budget in [1,4,16,64,256]:
            row=null_search(budget,80,2000,59+seed);row['seed']=seed;rows.append(row)
    frame=pd.DataFrame(rows);summary=[]
    for budget,g in frame.groupby('candidates'):
        gap=g.test_error-g.validation_error
        summary.append(dict(candidates=int(budget),validation_error=float(g.validation_error.mean()),test_error=float(g.test_error.mean()),
            optimism=float(gap.mean()),monte_carlo_se=float(gap.std(ddof=1)/np.sqrt(len(gap)))))
    return dict(scope='Synthetic pure-noise negative control, 200 independent trials',rows=rows,summary=summary,
        validation_rows=80,test_rows=2000,verdict='MECHANISM_ONLY',paper_reproduction='NOT_RUN')


def train_count_pfn(seed=0,steps=1500):
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed+61)
    model=CountPFN();opt=torch.optim.Adam(model.parameters(),lr=.003);history=[]
    for step in range(steps):
        # One probability per task; context AND query share it.
        sampled,query=sample_coin_tasks(rng)
        counts=torch.tensor(sampled,dtype=torch.float32)
        target=torch.tensor(query,dtype=torch.float32)
        loss=nn.functional.binary_cross_entropy_with_logits(model(counts),target)
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0:history.append(dict(step=step,query_loss=float(loss.detach())))
    model.eval();grid=[]
    with torch.no_grad():
        for n in [4,16,32,64]:
            counts=torch.tensor([[s,n-s] for s in range(n+1)],dtype=torch.float32)
            p=model(counts).sigmoid().numpy()
            for s,value in enumerate(p):grid.append(dict(n=n,successes=s,prediction=float(value),exact=posterior_predictive([1]*s+[0]*(n-s)),mle=s/n))
    return model,dict(seed=seed,steps=steps,history=history,grid=grid,
        in_support_mae=float(np.mean([abs(r['prediction']-r['exact']) for r in grid if r['n']<=32])),
        extrapolation_mae=float(np.mean([abs(r['prediction']-r['exact']) for r in grid if r['n']==64])))


def posterior_experiment():
    start=time.perf_counter();runs=[train_count_pfn(s)[1] for s in [0,1,2]]
    return dict(scope='From-scratch PFN objective on Beta(1,1)-Bernoulli tasks; no Transformer or TabPFN checkpoint',
        runs=runs,seconds=time.perf_counter()-start,verdict='MECHANISM_ONLY',
        training='Query-label BCE; analytic posterior used only for evaluation',paper_training='NOT_RUN')


def scm_experiment():
    rng=np.random.default_rng(63);noise=rng.normal(size=(1500,4))*.3
    w=np.array([[0.,1.4,0.,0.],[0.,0.,-1.5,1.],[0.,0.,0.,1.2],[0.,0.,0.,0.]])
    base=sample_scm(noise,w);changed=w.copy();changed[1,2]=0.;cut=sample_scm(noise,changed)
    return dict(scope='Paired synthetic intervention with identical exogenous noise',weights=w.tolist(),changed_weights=changed.tolist(),
        base=base[:200].tolist(),intervention=cut[:200].tolist(),
        downstream_mean_absolute_change=np.abs(base-cut).mean(0).tolist(),
        same_ancestors=bool(np.array_equal(base[:,:2],cut[:,:2])),verdict='MECHANISM_ONLY',
        limitation='Four-node tanh DAG, fixed observation choices; not the released TabPFN prior mixture')


def drift_experiment():
    # Same noise in each domain isolates a mechanism shift, not a sample-size effect.
    rng=np.random.default_rng(68);noise=rng.normal(size=(600,3));domains=np.arange(5)
    records=[]
    for domain in domains:
        w=np.zeros((3,3));w[0,1]=1.2;w[1,2]=1.5*np.cos(domain*.65)
        values=sample_scm(noise,w)
        records.append(dict(domain=int(domain),edge=float(w[1,2]),values=values[:120].tolist(),
            correlation=float(np.corrcoef(values[:,1],values[:,2])[0,1])))
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    # Train at t0/1, select on t2, report t3/4; compare pooled history vs recent window.
    full=[]
    for domain in domains:
        r=np.random.default_rng(680+domain);x=r.normal(size=(400,2));edge=1.5*np.cos(domain*.65)
        probability=1/(1+np.exp(-edge*x[:,1]));y=r.binomial(1,probability);full.append((x,y))
    results=[]
    for name,train_domains in [('pooled',[0,1]),('recent',[1])]:
        x=np.concatenate([full[t][0] for t in train_domains]);y=np.concatenate([full[t][1] for t in train_domains])
        candidates=[]
        for c in [.01,1.,100.]:
            m=LogisticRegression(C=c).fit(x,y);candidates.append((log_loss(full[2][1],m.predict_proba(full[2][0])),m,c))
        score,m,c=min(candidates,key=lambda r:r[0])
        for domain in [3,4]:results.append(dict(arm=name,domain=domain,selected_C=c,validation_loss=float(score),test_loss=float(log_loss(full[domain][1],m.predict_proba(full[domain][0])))))
    return dict(scope='Synthetic changing-edge diagnostic plus logistic negative control, not Drift-Resilient TabPFN training',
        domains=records,results=results,verdict='MECHANISM_ONLY',
        boundary='Paper uses a sampled second-order SCM and pretrained temporal PFN; this controlled cosine shift is a narrow generator intervention')


def temporal_tasks(rng,batch=16,drift=True):
    """A narrow second-order mechanism: time -> random tanh net -> one edge weight.

    Context times precede query times. Features contain [cause value, time]. The
    generated binary effect is the label. All parameters are sampled per task.
    """
    context=24;query=8
    times=np.concatenate([rng.uniform(0,.6,(batch,context)),rng.uniform(.6,1,(batch,query))],1)
    cause=rng.normal(size=(batch,context+query))
    a=rng.normal(size=(batch,1));b=rng.normal(0,2,(batch,1));c=rng.normal(0,2,(batch,1));d=rng.normal(size=(batch,1))
    weights=a+c*np.tanh(b*times+d) if drift else np.broadcast_to(a,times.shape)
    probability=1/(1+np.exp(-weights*cause));labels=rng.binomial(1,probability)
    x=np.stack([cause,times],-1).astype('float32')
    return torch.tensor(x[:,:context]),torch.tensor(labels[:,:context]),torch.tensor(x[:,context:]),torch.tensor(labels[:,context:])


def temporal_pfn_experiment(steps=400):
    """Controlled prior ablation, not the released Drift-Resilient TabPFN model."""
    torch.set_num_threads(1);rows=[];begin=time.perf_counter()
    for seed in [0,1,2]:
        for arm,drift in [('stationary-prior',False),('changing-edge-prior',True)]:
            torch.manual_seed(seed);rng=np.random.default_rng(6800+seed)
            model=RowPFN(2,width=16,heads=2,layers=2)
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            for _ in range(steps):
                cx,cy,qx,qy=temporal_tasks(rng,drift=drift)
                loss=nn.functional.cross_entropy(model(cx,cy,qx).reshape(-1,2),qy.reshape(-1))
                opt.zero_grad();loss.backward();opt.step()
            model.eval()
            for condition,shift in [('stationary',False),('drifting',True)]:
                # Same 200 evaluation tasks for every arm/seed, never used for optimization.
                cx,cy,qx,qy=temporal_tasks(np.random.default_rng(6899),batch=200,drift=shift)
                with torch.no_grad():
                    losses=nn.functional.cross_entropy(model(cx,cy,qx).reshape(-1,2),qy.reshape(-1),reduction='none').reshape(200,8).mean(1)
                rows.append(dict(seed=seed,arm=arm,condition=condition,log_loss=float(losses.mean()),task_losses=losses.tolist()))
    return dict(steps=steps,seconds=time.perf_counter()-begin,records=rows,
        scope='Two-layer RowPFN, matched compute, stationary versus sampled changing-edge prior; 200 held-out tasks per condition',
        verdict='MECHANISM_ONLY',paper_model='NOT_REPRODUCED')


def save_experiment(lesson,experiment):
    result=experiment();result['operator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/f'_verify_l{lesson:03}_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

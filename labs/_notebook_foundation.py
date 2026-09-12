"""Standalone notebook assembly with live student functions and executable checks."""
from _lesson_depth import enrich_notebook
import ast,base64,json,re,textwrap,html
from pathlib import Path
import nbformat as nbf
from _colab import bootstrap_cells
from _foundation_config import SLUGS,TITLES,PRIMARY,TASKS,PREDICT,ARCH

ROOT=Path(__file__).resolve().parent
SETUP='''# PROVIDED — environment, peripheral data/metric utilities
import os, sys, math, copy, re, json, hashlib, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata, friedmanchisquare, studentized_range, t
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from IPython.display import display
for directory in [Path.cwd(), Path.cwd()/'labs', Path.cwd().parent]:
    if (directory/'relkit').is_dir():
        ROOT=directory.resolve(); sys.path.insert(0,str(ROOT)); break
else:
    raise RuntimeError('Run the Colab bootstrap or start in the course repository')
torch.set_num_threads(1)
from relkit.foundation_benchmark import random_task, encode_train
print('CPU mechanism track; official checkpoint reruns use an isolated subprocess after EXIT.')
'''


def source(module,name):
    path=ROOT/'relkit'/f'{module}.py';text=path.read_text()
    return next(ast.get_source_segment(text,node) for node in ast.parse(text).body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)


CHECKS={
'parse_talent':'''frame=parse_talent(ROOT/'sources/l058/cls_bin.md')
assert len(frame)==120, 'Release coverage changed: inspect the pinned source before changing this expectation'
assert abs(frame.loc['Pima_Indians_Diabetes_Database','xgboost']-.7645)<1e-12, 'Mean parsing lost a sign, decimal or emphasis marker'
assert frame.index.is_unique, 'Do not hide duplicate dataset identities'
''',
'dataset_bootstrap':'''fixture=np.array([[1.,3.],[2.,2.],[3.,1.]])
interval=dataset_bootstrap(fixture,draws=2000,seed=58)
assert interval.shape==(2,2) and (interval[0]<=interval[1]).all()
np.testing.assert_allclose(interval[:,0],4-interval[::-1,1],atol=1e-12,
                           err_msg='Complementary columns must be sampled on the same dataset indices')
''',
'choose_validation':'''assert choose_validation([.4,.2,.3])==1
assert choose_validation([.2,.2,.3])==0, 'Declare deterministic tie handling'
for bad in ([],[.2,np.nan],[np.inf]):
    try: choose_validation(bad)
    except ValueError: pass
    else: raise AssertionError('Invalid validation vector accepted')
''',
'null_search':'''small=null_search(1,80,2000,59);large=null_search(256,80,2000,59)
assert large['validation_error']<=small['validation_error'], 'Nested candidate set cannot worsen the selected validation minimum'
assert abs(large['test_error']-.5)<.06, 'Noise-only test performance should remain near chance on this deterministic fixture'
assert large['selected']<256
''',
'validate_partitions':'''assert validate_partitions(dict(train=[0,1],val=[2],test=[3]))
for ids in [dict(train=[0,1],val=[2],test=[1,3]),dict(train=[0,0],val=[1],test=[2])]:
    try: validate_partitions(ids)
    except ValueError: pass
    else: raise AssertionError('Overlap or duplicate row IDs accepted')
''',
'paired_summary':'''fixture=[dict(dataset=d,arm=a,seed=s,error=e,seconds=1.)
 for d,errors in [('a',[1.,2.,3.]),('b',[3.,2.,1.])]
 for a,e in zip(['A','B','C'],errors) for s in range(3)]
summary=paired_summary(fixture)
assert all(v==2 for v in summary['mean_ranks'].values()), 'Average seeds first and preserve dataset weights'
try: paired_summary(fixture[:-1])
except ValueError: pass
else: raise AssertionError('An incomplete paired panel must not silently enter the ranking')
''',
'posterior_predictive':'''assert posterior_predictive([1,0,1])==.6
assert posterior_predictive([],2,3)==.4
assert posterior_predictive([1,0,1],2,3)==.5
try: posterior_predictive([2])
except ValueError: pass
else: raise AssertionError('Nonbinary labels accepted')
''',
'sample_coin_tasks':'''counts,query=sample_coin_tasks(np.random.default_rng(610),batch=10000,max_context=32)
assert counts.shape==(10000,2) and query.shape==(10000,)
assert (counts>=0).all() and ((counts.sum(1)>=1)&(counts.sum(1)<=32)).all()
cov=np.cov(counts[:,0]/counts.sum(1),query)[0,1]
assert cov>.04, 'Context and query must share theta; a fresh independent query task destroys their dependence'
''',
'context_mask':'''mask=context_mask(3,2)
assert mask.dtype==torch.bool and mask.shape==(5,5)
assert mask[:,:3].all() and not mask[:,3:].any(), 'Query columns must never supply context keys'
''',
'attention':'''torch.manual_seed(62)
q,k,v=torch.randn(2,3,4),torch.randn(2,5,4),torch.randn(2,5,6)
allowed=torch.tensor([[True,True,False,True,False]]).expand(3,5)
torch.testing.assert_close(attention(q,k,v,allowed),nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=allowed))
changed=v.clone();changed[:,2]=1000;changed[:,4]=-1000
torch.testing.assert_close(attention(q,k,v,allowed),attention(q,k,changed,allowed),msg='Blocked values changed the output')
''',
'sample_scm':'''noise=np.array([[1.,.2,-.1]])
w=np.array([[0.,2.,0.],[0.,0.,-1.],[0.,0.,0.]])
np.testing.assert_allclose(sample_scm(noise,w,lambda x:x),[[1.,2.2,-2.3]])
try: sample_scm(noise,w.T)
except ValueError: pass
else: raise AssertionError('This implementation requires topologically ordered weights')
''',
'observe_scm':'''values=np.array([[0.,1.,2.],[3.,4.,-1.]])
x,y=observe_scm(values,[0,1],2,threshold=0)
np.testing.assert_array_equal(y,[1,0]);np.testing.assert_array_equal(x,values[:,:2])
try: observe_scm(values,[0,2],2,0)
except ValueError: pass
else: raise AssertionError('Target included as a feature')
''',
'axial_sample_attention':'''torch.manual_seed(64);h=torch.randn(2,5,3,8)
block=AttentionBlock(8,2).eval();actual=axial_sample_attention(block,h,3)
expected=torch.stack([block(h[:,:,f,:],context_mask(3,2)) for f in range(3)],dim=2)
torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
changed=h.clone();changed[:,4]=1000
torch.testing.assert_close(axial_sample_attention(block,h,3)[:,:4],axial_sample_attention(block,changed,3)[:,:4],atol=1e-5,rtol=1e-5)
''',
'crossfit_embeddings':'''calls=[]
def diagnostic_embed(cx,cy,qx):
    calls.append((set(cx[:,0]),set(qx[:,0])));return qx*2
x=np.arange(12.).reshape(6,2);y=np.arange(6)%2;folds=np.array([0,1,0,1,0,1])
h=crossfit_embeddings(x,y,folds,diagnostic_embed)
np.testing.assert_array_equal(h,x*2)
assert all(not a&b for a,b in calls), 'A query entered its own labeled context'
''',
'induced_column':'''torch.manual_seed(66);u=torch.randn(7,8);inducing=torch.randn(3,8)
base=induced_column(u,inducing,4);changed=u.clone();changed[5:]=1000
torch.testing.assert_close(base[:5],induced_column(changed,inducing,4)[:5])
assert base.shape==u.shape
''',
'distribution_embedding':'''u=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
inducing=torch.eye(2);values=torch.tensor([2.,3.,4.])
out=distribution_embedding(values,u,inducing,2,lambda h:torch.ones_like(h)*.5,lambda h:torch.ones_like(h)*.1)
torch.testing.assert_close(out,torch.tensor([[1.1,1.1],[1.6,1.6],[2.1,2.1]]))
''',
'nearest_context':'''memory=np.array([[0.],[1.],[2.],[4.]])
ids=nearest_context(memory,np.array([[1.]]),2,np.array([1]))
assert ids.tolist()==[[0,2]], 'Exclude row identity; preserve deterministic equal-distance order'
duplicates=np.array([[1.],[1.],[2.]])
assert nearest_context(duplicates,duplicates[:1],1,np.array([0])).tolist()==[[1]], 'A distinct duplicate is still eligible'
''',
'local_episode':'''x=np.arange(40.).reshape(20,2);y=np.arange(20)%2
cx,cy,qx,qy,ci,qi=local_episode(x,y,10,4,3,np.random.default_rng(67))
assert len(ci)==4 and len(qi)==3 and not set(ci)&set(qi)
assert 10 not in ci and 10 not in qi
np.testing.assert_array_equal(cx,x[ci]);np.testing.assert_array_equal(qy,y[qi])
''',
'temporal_eligible':'''assert temporal_eligible([1,2,3],[2,8,4],4).tolist()==[True,False,True]
assert not temporal_eligible([5],[1],4)[0], 'An available label does not make a future event eligible'
''',
'temporal_tasks':'''cx,cy,qx,qy=temporal_tasks(np.random.default_rng(68),batch=8,drift=True)
assert cx.shape==(8,24,2) and qx.shape==(8,8,2)
assert float(cx[:,:,1].max())<float(qx[:,:,1].min()), 'Context domains must precede query domains'
assert set(cy.flatten().tolist())<= {0,1}
''',
'open_class_loss':'''p=np.array([[.8,.2],[.2,.8],[.1,.9]])
value=open_class_loss([0,1,2],p,[0,1])
expected=(-2*np.log(.8)-np.log(1e-12))/3
assert abs(value-expected)<1e-10, 'Unsupported rows must contribute to the all-row loss'
assert abs(open_class_loss([0,1],p[:2],[0,1])+np.log(.8))<1e-10
''',
'corrupt_column':'''train=np.array([[0.,1.],[2.,3.]])
test=np.array([[100.,5.]]);original=test.copy()
np.testing.assert_array_equal(corrupt_column(train,test,0,'missing'),[[1.,5.]])
np.testing.assert_array_equal(corrupt_column(train,test,0,'scale'),[[298.,5.]])
np.testing.assert_array_equal(test,original,err_msg='The clean baseline was mutated')
'''
}

COMMON_MODEL=[('foundation_core',v) for v in ['context_mask','attention','AttentionBlock','RowPFN']]
PROVIDED={58:[('foundation_experiments','survey_audit')],59:[('foundation_experiments','validation_audit')],
60:[('foundation_benchmark',v) for v in ['error','fit_candidate']],
61:[('foundation_core','CountPFN'),('foundation_experiments','train_count_pfn')],
62:COMMON_MODEL,63:[('foundation_experiments','scm_experiment')],
64:[('foundation_core',v) for v in ['attention','AttentionBlock','AxialPFN']],
65:COMMON_MODEL,66:[('foundation_core','attention')],67:COMMON_MODEL,
68:COMMON_MODEL,69:[],70:[]}

TRIAL={58:'''trial=survey_audit()
display(pd.DataFrame(trial['coverage']).T)
display(pd.DataFrame({'full':trial['mean_ranks'],'outcome_selected_45':trial['xgb_favored_45_ranks']}))
assert trial['datasets']==300
''',59:'''trial=validation_audit()
display(pd.DataFrame(trial['summary']))
assert trial['summary'][-1]['optimism']>trial['summary'][0]['optimism']
''',60:'''# Fresh five-arm smoke: your selector freezes choices before test scoring.
raw,y,reg,audit=random_task('diabetes',cap=240,seed=60);x=encode_train(raw)
validate_partitions(audit['ids'])
trial=[]
with threadpool_limits(limits=1):
    for arm in ['XGBoost','CatBoost','MLP','RealMLP-TD-S','TabM-mini']:
        candidates=[fit_candidate(arm,x['train'],y['train'],x['val'],y['val'],False,0,c,dict(trees=30,epochs=6)) for c in [0,1]]
        chosen=choose_validation([v[1] for v in candidates]);p=candidates[chosen][0](x['test'])
        trial.append(dict(arm=arm,selected=chosen,validation=[v[1] for v in candidates],test_log_loss=error(y['test'],p,False)))
display(pd.DataFrame(trial))
''',61:'''model,trial=train_count_pfn(seed=0,steps=1500)
print({k:trial[k] for k in ['seed','steps','in_support_mae','extrapolation_mae']})
assert trial['in_support_mae']<.12, 'Inspect the task generator and training, not the tolerance, if the posterior fit fails'
''',62:'''raw,y,reg,audit=random_task('diabetes',cap=120,seed=62);x=encode_train(raw)
scale=StandardScaler().fit(x['train']);cx=torch.tensor(scale.transform(x['train']),dtype=torch.float32)[None]
qx=torch.tensor(scale.transform(x['test'][:4]),dtype=torch.float32)[None];cy=torch.tensor(y['train'])[None]
torch.manual_seed(62);model=RowPFN(cx.shape[-1]).eval();p=model(cx,cy,qx)
torch.testing.assert_close(p[:,:1],model(cx,cy,qx[:,:1]),atol=1e-6,rtol=1e-5)
p.sum().backward();assert model.x_encoder.weight.grad.abs().sum()>0
trial={'shape':list(p.shape),'scope':'untrained real-row mechanism and gradient check, not pretrained accuracy'}
print(trial)
''',63:'''trial=scm_experiment()
assert trial['same_ancestors']
values=np.asarray(trial['base']);x,y=observe_scm(values,[0,1,2],3,0.)
print('Observed X/y:',x.shape,y.shape,'ancestor change:',trial['downstream_mean_absolute_change'][:2])
''',64:'''raw,y,reg,audit=random_task('diabetes',cap=120,seed=64);x=encode_train(raw)
scale=StandardScaler().fit(x['train']);cx=torch.tensor(scale.transform(x['train']),dtype=torch.float32)[None]
qx=torch.tensor(scale.transform(x['test'][:4]),dtype=torch.float32)[None];cy=torch.tensor(y['train'])[None]
torch.manual_seed(64);model=AxialPFN(cx.shape[-1],width=16,heads=2).eval()
prediction=model(cx,cy,qx);changed=torch.cat([qx,torch.ones_like(qx[:,:1])*100],1)
torch.testing.assert_close(prediction,model(cx,cy,changed)[:,:4],atol=2e-6,rtol=2e-5)
prediction.sum().backward();assert model.x_encoder.weight.grad.abs().sum()>0
trial={'shape':list(prediction.shape),'query_isolation':True,'scope':'untrained key-part architecture only'};print(trial)
''',65:'''raw,y,reg,audit=random_task('diabetes',cap=150,seed=65);x=encode_train(raw)
scale=StandardScaler().fit(x['train']);z={s:scale.transform(a).astype('float32') for s,a in x.items()}
torch.manual_seed(65);model=RowPFN(z['train'].shape[1],width=16,heads=2).eval()
def embed(cx,cy,qx):
    with torch.no_grad():return model.embeddings(torch.tensor(cx)[None],torch.tensor(cy)[None],torch.tensor(qx)[None])[0].numpy()
folds=np.arange(len(y['train']))%3
train_h=crossfit_embeddings(z['train'],y['train'],folds,embed)
head=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(train_h,y['train'])
test_h=embed(z['train'],y['train'],z['test']);p=head.predict_proba(test_h)
trial={'embedding_shape':list(train_h.shape),'test_log_loss':float(log_loss(y['test'],p)),
       'scope':'random-weight encoder diagnostic; actual pretrained v2 evidence is separate'};print(trial)
''',66:'''raw,y,reg,audit=random_task('phoneme',cap=120,seed=66);x=encode_train(raw)
values=torch.tensor(np.concatenate([x['train'][:,0],x['test'][:4,0]]),dtype=torch.float32)
torch.manual_seed(66);tokens=nn.Linear(1,8)(values[:,None]);inducing=torch.randn(3,8)
out=distribution_embedding(values,tokens,inducing,len(x['train']),nn.Linear(8,8),nn.Linear(8,8))
assert out.shape==(len(values),8);out.sum().backward()
trial={'cell_output_shape':list(out.shape),'context_rows':len(x['train']),'scope':'untrained distribution-aware column mechanism'};print(trial)
''',67:'''raw,y,reg,audit=random_task('diabetes',cap=150,seed=67);x=encode_train(raw)
scale=StandardScaler().fit(x['train']);z=scale.transform(x['train']).astype('float32')
torch.manual_seed(67);model=RowPFN(z.shape[1],width=16,heads=2);opt=torch.optim.Adam(model.parameters(),lr=.001)
rng=np.random.default_rng(67);losses=[]
for step in range(12):
    cx,cy,qx,qy,ci,qi=local_episode(z,y['train'],int(rng.integers(len(z))),24,8,rng)
    assert not set(ci)&set(qi)
    loss=nn.functional.cross_entropy(model(torch.tensor(cx)[None],torch.tensor(cy)[None],torch.tensor(qx)[None])[0],torch.tensor(qy))
    opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
trial={'episode_losses':losses,'scope':'small from-scratch episode smoke; actual v1 fine-tuning evidence is separate'};print(trial)
''',68:'''torch.manual_seed(68);model=RowPFN(2,width=16,heads=2);opt=torch.optim.Adam(model.parameters(),lr=.001)
rng=np.random.default_rng(68);losses=[]
for step in range(12):
    cx,cy,qx,qy=temporal_tasks(rng,batch=8,drift=True)
    assert float(cx[:,:,1].max())<float(qx[:,:,1].min())
    loss=nn.functional.cross_entropy(model(cx,cy,qx).reshape(-1,2),qy.reshape(-1))
    opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
trial={'eligible_ids':np.flatnonzero(temporal_eligible([1,2,3],[2,8,4],4)).tolist(),'losses':losses,
       'scope':'12-step temporal PFN smoke; complete matched prior ablation in author evidence'};print(trial)
''',69:'''raw,y,reg,audit=random_task('diabetes',cap=150,seed=69);x=encode_train(raw)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(x['train'],y['train'])
trial=[]
for condition in ['clean','missing','scale']:
    z=x['test'] if condition=='clean' else corrupt_column(x['train'],x['test'],0,condition)
    p=model.predict_proba(z)
    trial.append({'condition':condition,'all_row_loss':open_class_loss(y['test'],p,model.classes_)})
display(pd.DataFrame(trial));print('Fresh logistic diagnostic; author v2/XGB intervention is separate.')
''',70:'''evidence=json.loads((ROOT/'_verify_l070_results.json').read_text())
for row in evidence['records']:
    assert choose_validation(row['validation_errors'])==row['selected']
    assert abs(log_loss(row['targets'],row['predictions'],labels=[0,1])-row['error'])<1e-7
trial=paired_summary(evidence['records'])
display(pd.DataFrame(trial['details'])[['dataset','arm','mean','sd','seconds']])
requested=['XGBoost','TabM-mini','TabPFN-v2','TabICL-v1.1','TabPFN-2.5-synthetic','TabPFN-3','TabICLv2']
roster={arm:('MEASURED' if arm in trial['arms'] else 'NOT_RUN') for arm in requested}
assert all(v=='MEASURED' for v in roster.values()), 'The seven-arm checkpoint requires every declared version'
assert len(evidence['records'])==105
print('Complete seven-arm panel:',trial['mean_ranks']);print('Requested-arm ledger:',roster)
'''}


def build(n,solution=False):
    if n == 64:
        from _build_l064 import build as build_l064
        return build_l064(solution)
    if n == 66:
        from _build_l066 import build as scoped_build
        return scoped_build(solution)
    if n == 65:
        from _build_l065 import build as build_l065
        return build_l065(solution)
    if n == 63:
        from _build_l063 import build as build_l063
        return build_l063(solution)
    if n == 62:
        from _build_l062 import build as build_l062
        return build_l062(solution)
    if n == 61:
        from _build_l061 import build as build_l061
        return build_l061(solution)
    if n == 60:
        from _build_l060 import build as build_l060
        return build_l060(solution)
    if n == 59:
        from _build_l059 import build as build_l059
        return build_l059(solution)
    if n == 58:
        from _build_l058 import build as build_l058
        return build_l058(solution)
    slug=f'{n:04}-{SLUGS[n]}';cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):cells.append(nbf.v4.new_code_cell(text.strip()))
    def figure(name):
        path=ROOT/f'figures/l{n:03}/{name}.png'
        if not path.exists():raise FileNotFoundError(path)
        alt=f'L{n:03} {name}: '+('author-reference measurements, separate from your kernel' if name in ['results','comparison','ranks'] else 'worked computation; see adjacent explanation')
        md(f'<div style="overflow-x:auto;max-width:100%"><img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" style="width:900px;min-width:640px;max-width:100%;height:auto"></div>\n\n*{alt}. Scroll wide figures horizontally on small screens.*')
    md(f'# Lab {n:03} · {TITLES[n]}\n\n[Lesson](../lessons/{slug}.html) · [Reference](../reference/{slug}.html) · [Reproduction contract](l{n:03}-reproduction.md)\n\n**Route:** retrieve → explain → predict → implement → CHECK → inspect evidence → EXIT. PROVIDED cells are readable implementation and peripheral support. TODOs contain your load-bearing code; CHECKs give immediate diagnostic feedback. Complete the code before revealing teacher solutions. Core study: 35–50 minutes; lab: 45–75 minutes plus explicitly gated larger runs.\n\n**Before reading:** {PREDICT[n][0]} Write your answer and justify the information boundary. Then recall one older lesson that could expose an error in this experiment.\n\n**Evidence contract:** numerical figures labeled author-reference come from committed author measurements. They are not outputs of your current kernel. Reduced model mechanisms, pretrained reference inference, frozen-result reanalysis and paper reproduction are different tracks. Every track states its scope below.')
    for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
    text=(ROOT.parent/'lessons/content'/f'{slug}.md').read_text()
    for part in re.split(r'(<!--figure:\w+-->)',text):
        match=re.fullmatch(r'<!--figure:(\w+)-->',part)
        if match:figure(match.group(1))
        elif part.strip():md(part)
    md('## Reproduction contract and task support\n\nThe visible code below implements the named operator or reduced architecture. The full pretrained model is a separately pinned reference path. Real public-table experiments use Tier A data; synthetic coin/SCM/shift tasks are Tier C mechanism isolation. Seeds, row identities, recipe budgets, versions and source/checkpoint hashes are available in the linked contract and evidence. Numerical agreement with a local operator is not a paper-table reproduction.')
    code(SETUP)
    task_names={v[0] for v in TASKS[n]}
    if n==60:
        code('# PROVIDED — peripheral tree implementations\nfrom xgboost import XGBClassifier,XGBRegressor\nfrom catboost import CatBoostClassifier,CatBoostRegressor')
        for module in ['realmlp','tabm']:
            text=(ROOT/'relkit'/f'{module}.py').read_text()
            for node in ast.parse(text).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                    code('# PROVIDED — visible reused L053/L054 model component\n'+ast.get_source_segment(text,node))
    # Functions may reference student names before definition, but no forward call happens yet.
    for module,name in PROVIDED[n]:
        if name not in task_names:code(f'# PROVIDED — {name}; inspect the live function calls\n'+source(module,name))
    for i,(name,module,signature,goal) in enumerate(TASKS[n],1):
        md(f'## TODO {i} · {name}\n\n**Goal:** {goal}\n\n**Why it matters:** a wrong implementation can produce a plausible-looking experiment while violating this lesson\'s prediction contract. Before coding, name one input that should fail or one intervention that should leave the result unchanged.\n\n**Hint boundary:** reason from the worked example and the function contract; the CHECK verifies behavior, not a required code layout.')
        code(f'# TODO {i} — {name}\n'+(source(module,name) if solution else signature+'\n    raise NotImplementedError("Implement the stated operation")'))
        code(f'# CHECK {i} — {name}\n'+CHECKS[name]+f'\nprint("PASS: {name}")')
    md('## Run your live implementation\n\nPredict the outcome before execution. This cell calls the functions you just wrote. Changing a TODO must change the corresponding computation or make a CHECK fail. The text printed with the result identifies whether this is a small training run, a mechanism test or a reanalysis of committed measurements.')
    code('# PROVIDED — experiment driven by your live functions\n'+TRIAL[n])
    md('## Audit author-reference evidence\n\nInspect the measured results below independently of your smoke experiment. A fresh smoke may use a smaller budget. Preserve the protocol difference rather than forcing numerical agreement with a different run. The committed prediction arrays let you reconstruct scores without downloading pretrained weights.')
    code(f'''# PROVIDED — readable evidence, never impersonating current-kernel training
author=json.loads((ROOT/'_verify_l{n:03}_results.json').read_text())
print('AUTHOR REFERENCE:',author.get('scope',author.get('note','Declared checkpoint experiment')))
print('Evidence verdict:',author.get('verdict'))
rows=author.get('records',author.get('summary',[]))
if isinstance(rows,list) and rows:
    display(pd.DataFrame(rows).drop(columns=['predictions','targets','history','context_ids','task_losses'],errors='ignore').head(24))
if {n} in [60,70]:
    for regime,summary in author['summary'].items():
        print(regime,summary['mean_ranks'],'Friedman p=',summary['friedman_p'],'CD=',summary['nemenyi_cd'])
''')
    md('## EXIT TICKET · submit an executable argument\n\nProvide your code output plus a 30–100 word verdict: which implementation ran, what information it could use, what evidence supports the result, and which paper claim remains untested. Include an unexpected or failed case and a next discriminating experiment. The length check below only enforces an attempt; the tutor grades your reasoning. Do not infer personal mastery from a green code cell.')
    verdict=(PREDICT[n][1]+' This notebook verifies the declared local computation and audits separate author evidence. It does not establish reproduction of the full published training or evaluation protocol. I would test the stated information boundary on a fresh, prespecified evaluation before making a broader claim.') if solution else ''
    code(f'''# TODO — written verdict; paste the resulting artifact and explanation to the tutor
verdict={verdict!r}
assert len(verdict.split())>=30, 'Write a defensible verdict; a passing code cell is not the explanation'
ticket={{'lesson':{n},'checks':'passed in this kernel','trial':trial,'verdict':verdict,
        'paper_reproduction':'NOT_ESTABLISHED','author_reference':'_verify_l{n:03}_results.json'}}
folder=ROOT/'data/cache/foundation';folder.mkdir(parents=True,exist_ok=True)
path=folder/'student-l{n:03}-exit.json'
path.write_text(json.dumps(ticket,indent=2,default=lambda x:x.item() if hasattr(x,'item') else x.tolist(),allow_nan=False)+'\\n')
print('EXIT artifact:',path)
''')
    md(f'## NEXT STEP · reproduce or scale the declared track\n\nRead [the exact reproduction contract](l{n:03}-reproduction.md) before increasing resources. The gate below invokes the shared runner in a subprocess; historical versions are installed into isolated package directories so v1 and v2 cannot silently replace one another. It may download the pinned checkpoint. Larger runs remain INCOMPARABLE wherever material paper-protocol gaps remain. A full paper preset deliberately fails until the original protocol is implemented.\n\nFor unattended CPU execution: `modal run --detach modal/foundation_repro.py --lesson {n} --preset closer`. This is a supplied operator, not evidence that a cloud job was run.')
    code(f'''# PROVIDED — explicit post-EXIT gate (OFF by default)
RUN_PAPER_REPRO=False
if RUN_PAPER_REPRO:
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'_run_foundation.py'),'--lesson','{n}',
                    '--preset','closer','--output',str(ROOT/'data/cache/foundation/l{n:03}-closer.json')],check=True)
else:
    print('Scale-up NOT_RUN in this kernel. See the protocol ledger before enabling.')
''')
    nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},'lesson':n,'evidence_boundary':'operator vs pretrained inference vs paper reproduction'}),n)
    # Stable cell IDs keep regeneration reviewable.
    import hashlib
    for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{slug}:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
    out=ROOT/('solutions' if solution else '')/f'{slug}.ipynb';out.parent.mkdir(exist_ok=True);nbf.write(nb,out)
    return out

if __name__=='__main__':
    for n in range(58,71):build(n);build(n,True)

"""Build student and teacher notebooks with visible, live checkpoint implementation."""
import base64
import json
from pathlib import Path
import nbformat as nbf
from bs4 import BeautifulSoup
from _build_l047 import extract
from _build_l049 import mdtext
from _colab import bootstrap_cells
HERE=Path(__file__).resolve().parent;SLUG='0050-q1-checkpoint'
SOURCE=(HERE/'relkit/checkpoint.py').read_text()
LESSON=BeautifulSoup((HERE.parent/'lessons'/f'{SLUG}.html').read_text(),'html.parser')

def build(solution=False):
    cells=[]
    def md(x):cells.append(nbf.v4.new_markdown_cell(x.strip()))
    def code(x):cells.append(nbf.v4.new_code_cell(x.strip()))
    def section(key):md(mdtext(LESSON.find('section',id=key)))
    def provided(names,why):code('# PROVIDED — '+why+'\n'+extract(SOURCE,set(names)))
    def task(name,student):code('# TODO — teacher answer\n'+extract(SOURCE,{name}) if solution else student)
    def figure(name,caption):
        data=base64.b64encode((HERE/'figures/l050'/f'{name}.png').read_bytes()).decode()
        md(f'![{caption}](data:image/png;base64,{data})\n\n{caption}')
    md('''# Lab 050 · Q1 checkpoint
## FT-Transformer versus XGBoost under one protocol

[Lesson](../lessons/0050-q1-checkpoint.html) · [Quick reference](../reference/fair-comparison-checkpoint.html)

**Your skill:** construct a defensible comparison and keep your conclusion within its evidence. PROVIDED means readable implementation; TODO is your work; CHECK gives immediate diagnostics; EXIT combines output with your written claim audit.

**Scope:** full numeric FT-Transformer, XGBoost and MLP on three real OpenML tasks. This reviews an existing model, strengthens its implementation fidelity, and tests your protocol discipline. No categorical, regression, multiclass or ensemble branch. Local results are INCOMPARABLE to the paper benchmark. Required next step: a larger Higgs Small attempt with a printed ledger.

**Environment:** CPU sufficient for the learning lab (author run ~91 seconds; implementation and reading take longer). Colab bootstrap is the first code cell. PNG figures are embedded and need no execution. Author-reference snapshots are labeled; they are not your kernel output.

**Recall without opening L049:** why can exact model-output parity coexist with an INCOMPARABLE benchmark number? Write two sentences before proceeding.''')
    for c in bootstrap_cells():
        cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
    section('question')
    code('''# PROVIDED — environment and data harness; no imported model is trained.
import os, sys, copy, hashlib, inspect, json, time, marshal
from pathlib import Path
import importlib.metadata
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import t, rankdata, friedmanchisquare, studentized_range, binomtest
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
thread_limit=threadpool_limits(limits=1)
torch.set_num_threads(1)
for candidate in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (candidate/'relkit').is_dir():
        LABS=candidate.resolve();os.chdir(LABS);sys.path.insert(0,str(LABS));break
else: raise RuntimeError('Run from the repository root, labs/, or labs/solutions/.')
from relkit.data import load_tier_a, CACHE, SPECS
# The harness records the canonical source hash; EXIT adds live code identity.
__file__=str(LABS/'relkit/checkpoint.py')
print('Working directory:',LABS)''')
    section('protocol');figure('fit-boundary','Synthetic held-out intervention; the training-only estimate stays fixed.')
    md('''### Task 1 · Fit the information boundary
**Goal:** implement median imputation followed by standardization using only `train` indices.
**Why:** a split does not protect you if preprocessing fitted on held-out rows already.
**Hint boundary:** compute median, fill missing values, then compute training mean and population SD; use scale 1 for constants. Return transformed float32 data and JSON-compatible fit parameters. Predict the held-out intervention CHECK before running it.''')
    task('prepare_numeric','''# TODO 1 — complete the training-only fit.
def prepare_numeric(x, train):
    x=np.asarray(x,dtype=float)
    median = ____
    if not np.isfinite(median).all():
        raise ValueError('Entirely missing training feature.')
    filled=np.where(np.isnan(x),median,x)
    mean,scale = ____, ____
    scale=np.where(scale==0,1,scale)
    return ____, {'median':median.tolist(),'mean':mean.tolist(),'scale':scale.tolist()}''')
    code('''# CHECK 1 — change held-out rows, retain the same training transformation.
x=np.array([[0.,5.],[2.,5.],[np.nan,7.],[100.,9.]])
z,state=prepare_numeric(x,np.array([0,1]))
x[2:]=10000
+z2,state2=prepare_numeric(x,np.array([0,1]))
assert state==state2, 'Held-out rows changed fitted parameters.'
assert np.allclose(z[:2,0],[-1,1]) and np.array_equal(z[:2],z2[:2])
assert np.isfinite(z).all() and np.array_equal(z[:2,1],[0,0]), 'Constant/missing feature policy failed.'
print('CHECK 1 passed: held-out intervention cannot affect training preprocessing.')'''.replace('\n+z2','\nz2'))
    section('model');figure('architecture','Numeric FT-T, d=32/two blocks/four heads. First attention LayerNorm skipped; ReGLU shown inside the block.')
    md('''### Task 2 · Reconstruct the gated update
**Goal:** implement ReGLU from its two equal halves.
**Why:** L046 used GELU; preserving a model name while changing its activation does not establish exact model fidelity.
**Hint boundary:** gate the first half with the rectified second half; preserve all leading dimensions. The CHECK examines gradients as well as values.''')
    task('reglu','''# TODO 2 — Appendix E feed-forward activation.
def reglu(x):
    a,b = ____
    return ____''')
    code('''# CHECK 2 — values and derivatives distinguish the two halves.
x=torch.tensor([[2.,-1.,3.,-4.]],requires_grad=True)
y=reglu(x)
assert torch.equal(y,torch.tensor([[6.,0.]])), 'Use a * ReLU(b), not ReLU(a) * b.'
y.sum().backward()
assert torch.equal(x.grad,torch.tensor([[3.,0.,2.,0.]])), 'The closed gate must block both routes.'
print('CHECK 2 passed.')''')
    provided(['CheckpointAttention'],'§3.3 attention, explicit Q/K/V and weighted sums')
    provided(['CheckpointBlock'],'Appendix E: residual wiring and your live ReGLU')
    provided(['CheckpointFT','CheckpointMLP'],'complete numeric model and simple neural baseline')
    provided(['reference_parity'],'copied-weight validation against the installed reference, not a library training arm')
    code('''# CHECK — the reference is a validator; your model is the measured implementation.
parity=reference_parity()
print(parity)
original_reglu=reglu
calls=[]
def counted_reglu(x):
    calls.append(tuple(x.shape));return original_reglu(x)
reglu=counted_reglu
probe=CheckpointFT(4,d=16,layers=2,heads=4)
probe(torch.randn(3,4)).sum().backward()
reglu=original_reglu
assert len(calls)==2,'The visible model bypassed your ReGLU.'
print('CHECK: the student activation lies on the forward/backward path.')''')
    section('selection');figure('selection','Synthetic candidates: validation picks B; choosing A from its test score invalidates the evaluation.')
    md('''### Task 3 · Select without access to test
**Goal:** choose the first candidate attaining maximum finite validation AUROC.
**Why:** the function signature itself should exclude test scores.
**Hint boundary:** reject an empty, non-vector or non-finite input. Return an integer candidate index. The synthetic test scores in the explanation are deliberately absent from this function.''')
    task('select_trial','''# TODO 3 — validation-only selection.
def select_trial(validation_scores):
    scores=np.asarray(validation_scores)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Need finite validation scores.')
    return ____''')
    code('''# CHECK 3 — maximum and declared tie rule.
assert select_trial([.80,.86])==1, 'Candidate B wins validation.'
assert select_trial([.86,.86])==0, 'Exact ties choose the first candidate.'
try: select_trial([.8,np.nan])
except ValueError: pass
else: raise AssertionError('Non-finite validation scores must fail.')
print('CHECK 3 passed.')''')
    provided(['train_neural'],'read early stopping, best-weight restoration, loss and optimizer groups')
    provided(['predict'],'batched inference; XGBoost uses its best iteration')
    section('uncertainty')
    md('''### Task 4 · Pair before summarizing
**Goal:** return the mean difference, sample SD and conditional 95% t interval.
**Why:** separate error-bar overlap is not a paired comparison.
**Hint boundary:** subtract corresponding seeds first; use `ddof=1` and S−1 degrees of freedom. A one-seed smoke run should return `None` for SD and interval, not fabricate uncertainty.''')
    task('paired_summary','''# TODO 4 — paired conditional uncertainty.
def paired_summary(a,b):
    delta = ____
    if len(delta)<2:
        return {'mean':float(delta.mean()),'sd':None,'ci95':None}
    mean,sd = ____, ____
    half = ____
    return {'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}''')
    code('''# CHECK 4 — parallel score shifts and a nontrivial interval.
s=paired_summary([.80,.82,.84],[.79,.81,.83])
assert np.isclose(s['mean'],.01) and s['sd']<1e-12
s=paired_summary([.80,.85,.90],[.80,.80,.80])
assert np.isclose(s['sd'],.05)
assert np.allclose(s['ci95'],[.05-4.3026527299*.05/np.sqrt(3),.05+4.3026527299*.05/np.sqrt(3)])
print('CHECK 4 passed: intervals quantify training-seed variation only.')''')
    md('''### Read the whole experiment before running
The loader is reusable infrastructure. The loop below creates your visible `CheckpointFT`, calls your preprocessing and selector, and computes test probabilities only after candidate selection. `model_factory`, `prepare` and `selector` can be inspected at the call boundary. Two candidates × three seeds × three arms × three tasks = 54 fitted candidates. A seed's checkpoint/candidate selection both use validation; this is a low-budget protocol, not nested cross-validation.''')
    provided(['run_comparison'],'complete selection-and-test loop; uses the functions you just wrote')
    md('''### Predict first
Write which task you expect to show the largest tree/neural difference, and why. Distinguish that conjecture from a source-backed paper claim. Do not change candidate lists after seeing the following author-reference results.''')
    section('results');figure('scores','Author-reference AUROC; individual seeds and sample SD. These are saved reference measurements.')
    figure('paired','Author-reference paired differences; 95% t intervals conditional on one split.')
    code('''# PROVIDED — run YOUR implementation; no stored result substitutes for training.
result=run_comparison(model_factory=CheckpointFT,prepare=prepare_numeric,selector=select_trial)
rows=[]
for dataset,row in result['datasets'].items():
    for model,s in row['summary'].items():
        rows.append({'dataset':dataset,'model':model,'mean AUROC':s['mean'],'sample SD':s['sd'],'seed CI':s['ci95']})
display(pd.DataFrame(rows))
display(pd.DataFrame({name:row['ft_minus_xgb'] for name,row in result['datasets'].items()}).T)
print('Ranks:',result['mean_ranks'])
print('Dataset-level statistics:',result['statistics'])''')
    code('''# CHECK — audit saved selection and recompute every test metric.
for dataset,row in result['datasets'].items():
    split=row['split'];sets=[set(split[k]) for k in ('train','valid','test')]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]), 'Overlapping split rows.'
    for model,runs in row['runs'].items():
        for run in runs:
            assert run['selected_trial']==select_trial([v['validation_auc'] for v in run['trials']])
            assert np.isclose(run['auc'],roc_auc_score(row['y_test'],run['probability']))
assert result['statistics']['independent_units']==3
reference=json.loads((LABS/'_verify_l050_results.json').read_text())
max_gap=max(abs(run['auc']-reference['datasets'][d]['runs'][m][i]['auc'])
    for d,row in result['datasets'].items() for m,runs in row['runs'].items() for i,run in enumerate(runs))
print('CHECK: splits, choices and metrics audit clean. Largest author-reference gap:',max_gap)
print('Different software/hardware may change training; investigate rather than overwrite scores.')''')
    if solution:code("# Teacher-only verification: this execution uses the recorded local environment.\nassert max_gap < 1e-10, 'Teacher rerun differs from recorded evidence.'")
    md('''## EXIT · defend the result
Export the score table, paired intervals, rank summary and run ledger. Write 120–180 words naming the population, metric, candidate budget, uncertainty unit, one measured finding, and at least two paper-protocol mismatches. Propose one new experiment whose settings you would freeze before viewing results. Do not equate nonsignificance with equivalence or “won two tasks” with universal superiority.

Paste these outputs and your explanation to the teacher for a 0–10 review: correctness, leakage discipline, conceptual explanation, independent implementation effort, reproduction audit. A defensible INCOMPARABLE result can earn full audit credit; an unrun next step must remain NOT_RUN.''')
    code('''# EXIT — save evidence and identify the code objects that actually ran.
live_objects=[prepare_numeric,reglu,select_trial,paired_summary,train_neural,run_comparison,
              CheckpointFT.forward,CheckpointBlock.forward,CheckpointAttention.forward]
implementation_id=hashlib.sha256(b''.join(marshal.dumps(f.__code__) for f in live_objects)).hexdigest()
result['visible_code_identity']=implementation_id
Path('l050-student-results.json').write_text(json.dumps(result,indent=2))
print('Saved l050-student-results.json; implementation identity',implementation_id)
# Write your claim audit in the next markdown cell.''')
    md('''**My claim audit:** _Write your 120–180 words here. This prose is reviewed by the teacher, not automatically graded._''')
    section('scale')
    md('''### Required next step · choose compute, preserve the claim boundary
The same visible experiment runs on Higgs Small. The imported operator only handles presets, per-seed persistence and the conclusion ledger; your visible trainer and model remain in `run_comparison`. The gate is off by default. For Colab, select a GPU runtime before enabling it. Completion of the larger experiment is distinct from passing the local implementation CHECKs.

The fingerprint below distinguishes your code from the canonical operator's cache. Choose a new output folder when the code, environment or data changes. The `paper` preset remains an incomplete protocol reconstruction and must not be called MATCH.''')
    code('''# PROVIDED — larger-run gate; no cloud job launches automatically.
from _paper_repro_l050 import run as scaleup_operator, TARGET
from relkit.paper_repro import format_ledger, LabFinding
RUN_PAPER_REPRO=False
PAPER_PRESET='closer'
if RUN_PAPER_REPRO:
    def live_experiment(**kwargs):
        return run_comparison(model_factory=CheckpointFT,prepare=prepare_numeric,selector=select_trial,**kwargs)
    scaleup=scaleup_operator(PAPER_PRESET,out='data/cache/l050-student-'+implementation_id[:12],
        device='cuda' if torch.cuda.is_available() else 'cpu',experiment=live_experiment,
        implementation_id=implementation_id)
else:
    print(format_ledger(title='L050 student session',
        lab=[LabFinding('Your completed local experiment',str(result['mean_ranks']),
                        '3 numeric substitute datasets; two candidates; fixed split')],
        paper=[(TARGET,None,'NOT_RUN')]))''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'lesson':50})
    return nb

if __name__=='__main__':
    for solution in (False,True):
        path=HERE/('solutions' if solution else '')/f'{SLUG}.ipynb'
        nbf.write(build(solution),path);print(path)

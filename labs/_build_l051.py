"""Self-contained explanatory notebook; four live student intervention functions."""
import ast,base64,json,os
from urllib.parse import urlsplit
from pathlib import Path
import nbformat as nbf
from bs4 import BeautifulSoup
from _build_l047 import extract
from _build_l049 import mdtext
from _colab import bootstrap_cells
HERE=Path(__file__).parent;SLUG='0051-why-trees-still-win'
SOURCE=(HERE/'relkit/bias_interventions.py').read_text();MODEL=(HERE/'relkit/checkpoint.py').read_text()
LESSON=BeautifulSoup((HERE.parent/'lessons'/f'{SLUG}.html').read_text(),'html.parser')
def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 def section(name):
    node=BeautifulSoup(str(LESSON.find('section',id=name)),'html.parser').find('section')
    for link in node.find_all('a',href=True):
        parts=urlsplit(link['href'])
        if parts.scheme or parts.netloc or not parts.path:continue
        target=(HERE.parent/'lessons'/parts.path).resolve()
        link['href']=os.path.relpath(target,HERE)+('?' + parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
    md(mdtext(node))
 def provided(names,why,source=SOURCE):code('# PROVIDED — '+why+'\n'+extract(source,set(names)))
 def task(name,student):code('# TODO — teacher answer\n'+extract(SOURCE,{name}) if solution else student)
 def figure(name,caption):md('!['+caption+'](data:image/png;base64,'+base64.b64encode((HERE/'figures/l051'/f'{name}.png').read_bytes()).decode()+')\n\n'+caption)
 md('''# Lab 051 · Why trees still win: re-derive the biases

[Lesson](../lessons/0051-why-trees-still-win.html) · [Reference](../reference/inductive-bias-interventions.html)

**Skill:** design and implement an intervention that distinguishes information content from learning behavior. **Scope: key parts** of Grinsztajn et al. §5; familiar numeric MLP and FT-T from L050 are displayed in full. XGBoost is a packaged baseline. You write four live functions, then the visible harness calls them in the actual experiment.

**Route:** recall → smoothing → rotation → noise → paired effects → complete training code → three-task run → EXIT → required larger run. PROVIDED is read/run; TODO is your implementation; CHECK gives diagnostics. Author-reference figures are measured snapshots, separate from your current outputs.

**Data contract:** Tier A, electricity 44120, MagicTelescope 44125 and bank-marketing 44126 from the authors' January 2023 OpenML suite 337. The learning lab uses 1800 rows per task, 60/20/20 stratified split seed 51, model seeds 0/1/2, fixed intervention seed 51. The suite is a later release than the 2022 paper v1 read here. Numeric binary classification only. Three tasks and fixed recipes do not reproduce the benchmark search curves: **INCOMPARABLE**.

Computation took about two CPU minutes in the author run; understanding and implementation take longer. Six portable inline PNGs require no execution. Local packaging checks do not certify the live Colab UI.

**Recall first:** why did three seeds in L050 not create three new datasets? Write your answer before proceeding.''')
 cells.extend(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']) for c in bootstrap_cells())
 code('''# PROVIDED — environment; local repository root, labs/, or labs/solutions/ supported.
import os,sys,copy,hashlib,json,time,marshal,importlib.metadata,platform,types
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import special_ortho_group,t,rankdata,friedmanchisquare,studentized_range
from scipy.spatial.distance import cdist
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import MinCovDet
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.datasets import fetch_openml
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from IPython.display import display
import matplotlib.pyplot as plt
thread_limit=threadpool_limits(limits=1);torch.set_num_threads(1)
for candidate in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (candidate/'relkit').is_dir():
        LABS=candidate.resolve();sys.path.insert(0,str(LABS));os.chdir(LABS);break
else:raise RuntimeError('Run inside the course repository or use Colab bootstrap')
__file__=str(LABS/'relkit/bias_interventions.py')
print('Working directory:',LABS)''')
 section('question');section('smooth');figure('smoothing','Synthetic smoothing: self-weight, neighbor weights, probability and hard threshold; test targets stay fixed.')
 md('''### TODO 1 · Reconstruct the training-only smoother
Implement `smooth_targets(x,y,h,covariance)`. The covariance is already fitted on training rows. Chunking only limits memory; all training neighbors remain eligible. For h=0 copy the labels. For h>0 calculate the inverse-covariance distance, normalized weighted average and strict binary threshold. The three-row example is one CHECK, not the only input your function must handle.

**Predict before running:** multiplying all feature values by 3 and the covariance by 9 should do what to the smoothed output?''')
 task('smooth_targets','''# TODO 1 — Appendix A.4.1 mechanism and released classification threshold.
def smooth_targets(x,y,h,covariance):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if h<0:raise ValueError('Lengthscale must be nonnegative')
    if h==0:return y.copy(),y.astype(int).copy()
    precision = ____
    probability=np.empty(len(y))
    for start in range(0,len(x),256):
        distances=cdist(x[start:start+256],x,'mahalanobis',VI=precision)**2
        weights = ____
        probability[start:start+256] = ____
    return probability, ____''')
 code('''# CHECK 1 — arithmetic, units, self-weight and threshold convention.
x=np.arange(3.)[:,None];y=np.array([0,1,0]);cov=np.eye(1)
p,labels=smooth_targets(x,y,1.,cov)
assert np.isclose(p[1],1/(1+2*np.exp(-.5))), 'Normalize by all weights, including self.'
assert np.array_equal(labels,[0,0,0]), 'Threshold the probability, not the individual weights.'
assert np.array_equal(smooth_targets(x,y,0,cov)[1],y)
assert np.allclose(smooth_targets(3*x,y,1,9*cov)[0],p), 'Covariance must compensate the units.'
assert np.array_equal(smooth_targets(np.zeros((2,1)),np.array([0,1]),1,cov)[1],[0,0]), 'Exactly .5 must map to 0.'
print('CHECK 1 passed')''')
 section('rotation');figure('rotation','Synthetic 45-degree rotation: same labels and invertible inputs, but the one-coordinate threshold becomes diagonal.')
 md('''### TODO 2 · Preserve a single coordinate system
Implement `rotate_splits(splits,rotation)`. The input is a list of arrays; return one transformed array for each. Reject a nonorthogonal matrix. Do not sample any new matrix inside this function.

**Prediction:** can the same dense first-layer activations be recovered exactly, even if a newly trained model later gets a different test score?''')
 task('rotate_splits','''# TODO 2 — apply one shared orthogonal transformation.
def rotate_splits(splits,rotation):
    r=np.asarray(rotation)
    if not ____:
        raise ValueError('Rotation must be orthogonal')
    return ____''')
 code('''# CHECK 2 — inverse, distances and transported neural weights.
r=special_ortho_group.rvs(4,random_state=8)
rng=np.random.default_rng(1);x=rng.normal(size=(7,4));v=rng.normal(size=(3,4));w=rng.normal(size=(5,4))
z,zv=rotate_splits([x,v],r)
assert np.allclose(z@r.T,x) and np.allclose(zv@r.T,v), 'All splits need the same inverse.'
assert np.allclose(z@(w@r).T,x@w.T), 'Weight transport preserves first-layer activations.'
try:rotate_splits([x],2*np.eye(4))
except ValueError:pass
else:raise AssertionError('Scaling by 2 is not an orthogonal transform.')
print('CHECK 2 passed; this is weight transport, not Adam training parity.')''')
 section('noise');figure('noise','Synthetic finite-sample noise search and MLP parameter accounting. No test-score claim follows from either plot alone.')
 md('''### TODO 3 · Add independent noise while preserving the original block
Implement `add_noise_features(splits,count,seed)`. Create one generator before processing the splits, append `count` N(0,1) columns to each array, and return the list. Labels are intentionally absent from the signature. Do not restart the generator inside the split loop.

**Written check:** why should you not assert exact zero sample correlation between generated columns and labels?''')
 task('add_noise_features','''# TODO 3 — independent draws, reproducible intervention.
def add_noise_features(splits,count,seed):
    rng = ____
    return ____''')
 code('''# CHECK 3 — original values, distinct splits, repeatability and zero-column identity.
x=np.arange(12).reshape(4,3);v=x+20
a,b=add_noise_features([x,v],5,51)
assert a.shape==(4,8) and np.array_equal(a[:,:3],x)
assert np.array_equal(b[:,:3],v) and not np.array_equal(a[:,3:],b[:,3:]), 'Do not reuse a noise block across splits.'
assert np.array_equal(a,add_noise_features([x,v],5,51)[0])
assert np.array_equal(add_noise_features([x],0,51)[0],x)
print('CHECK 3 passed')''')
 section('protocol');figure('protocol','Every condition preserves evaluation targets. Compare smoothing against top-five raw targets, not full-feature original.')
 md('''### TODO 4 · Pair the effect before estimating uncertainty
Implement `paired_effect(changed,baseline)`. Return mean, sample SD (`ddof=1`) and 95% t interval for seed-wise differences. With one seed return `None` for SD and interval. Positive means the intervention improved accuracy. These intervals condition on one split and one fixed intervention realization.

**Predict:** if all three paired differences are exactly +0.01, should the interval be wide just because the underlying model scores differ?''')
 task('paired_effect','''# TODO 4 — differences first; S model seeds means S−1 degrees of freedom.
def paired_effect(changed,baseline):
    delta = ____
    mean=float(delta.mean())
    if len(delta)<2:return dict(mean=mean,sd=None,ci95=None)
    sd = ____
    half = ____
    return dict(mean=mean,sd=sd,ci95=[mean-half,mean+half])''')
 code('''# CHECK 4 — pairing and uncertainty units.
e=paired_effect([.81,.86,.91],[.80,.85,.90])
assert np.isclose(e['mean'],.01) and e['sd']<1e-12
e=paired_effect([.8,.85,.9],[.8,.8,.8]);assert np.isclose(e['sd'],.05)
assert np.isclose(e['ci95'][1]-e['mean'],t.ppf(.975,2)*.05/np.sqrt(3))
assert paired_effect([.8],[.7])['ci95'] is None
print('CHECK 4 passed')''')
 md('''## PROVIDED · Familiar models, visible end to end
These are the L050 numeric models, not new architectures. An MLP maps [B,d] through two width-64 linear/ReLU/dropout layers to one logit. FT-T embeds each numeric feature as `x_j w_j + b_j`, prepends CLS, applies two attention/ReGLU residual blocks, normalizes CLS and reads one logit. The first block skips attention LayerNorm. B is rows, d is input columns; FT-T token width is 32. Sigmoid turns a logit into a probability. Read the implementations and locate where input dimensionality enters each one.''')
 provided(['reglu','CheckpointMLP'],'MLP and FT-T activation from L050',MODEL)
 provided(['CheckpointAttention','CheckpointBlock'],'attention, ReGLU and residual connections',MODEL)
 provided(['CheckpointFT'],'complete numeric FT-T prediction path',MODEL)
 provided(['fit_arm'],'fixed recipe, validation loss, restored checkpoint and test inference')
 constants='\n'.join(ast.get_source_segment(SOURCE,n) for n in ast.parse(SOURCE).body if isinstance(n,ast.Assign))
 code('# PROVIDED — declared datasets, presets, conditions and model arms\n'+constants)
 provided(['load_task'],'load exact OpenML IDs, retain file hash and target mapping')
 provided(['prepare_task'],'read training-only feature ranking, covariance, and calls to your live interventions')
 provided(['rank_summary','run_experiment'],'full experiment and dataset-level rank statistics')
 code('''# CHECK — prove your functions are on the real preparation path.
originals={n:globals()[n] for n in ['smooth_targets','rotate_splits','add_noise_features']};calls={n:0 for n in originals}
def counted(name,fn):
    def wrapped(*args,**kwargs):
        calls[name]+=1;return fn(*args,**kwargs)
    return wrapped
try:
    for name,fn in originals.items():globals()[name]=counted(name,fn)
    states,labels,metadata=prepare_task('electricity',LABS/'data/cache/l051',500)
finally:
    globals().update(originals)
assert all(calls[n]==1 for n in calls), 'The preparation path bypassed a student function.'
assert np.array_equal(states['smoothed'][0][0],states['top5'][0][0]), 'Smoothing changed the feature set.'
assert np.array_equal(states['smoothed'][0][2],states['top5'][0][2])
assert np.array_equal(labels[2],metadata['test_y'])
print('CHECK: live intervention calls:',calls)''')
 md('''## Predict before measuring
Write one expected sign for each model under each intervention, plus one reason the sign might differ on a small task. Then run the complete comparison. The author-reference snapshot comes afterward; do not edit seeds or conditions until your result matches it.''')
 code('''# PROVIDED — trains visible models and uses your current student functions.
result=run_experiment('lab',cache=LABS/'data/cache/l051')
records=[]
for dataset,row in result['datasets'].items():
    for condition,runs in row['runs'].items():
        for model,rr in runs.items():
            a=np.array([r['accuracy'] for r in rr])
            records.append(dict(dataset=dataset,condition=condition,model=model,accuracy=a.mean(),seed_sd=a.std(ddof=1)))
scores=pd.DataFrame(records);display(scores.round(4))
display(pd.DataFrame(result['statistics']).T)
print('INCOMPARABLE to paper search curves; seconds:',round(result['elapsed_s']))''')
 code('''# PROVIDED — YOUR runtime effects with paired seed points and conditional intervals.
fig,axes=plt.subplots(3,1,figsize=(8,10),layout='constrained')
for ax,condition in zip(axes,['rotated','noise','smoothed']):
    for i,(dataset,row) in enumerate(result['datasets'].items()):
        for j,model in enumerate(MODELS):
            e=row['effects'][condition][model];pos=i+(j-1)*.2
            base='top5' if condition=='smoothed' else 'original'
            d=np.array([a['accuracy']-b['accuracy'] for a,b in zip(row['runs'][condition][model],row['runs'][base][model])])*100
            color=['#276e58','#276899','#ad4c35'][j]
            ax.scatter(d,[pos]*len(d),color=color,s=16)
            ax.errorbar(e['mean']*100,pos,xerr=[[100*(e['mean']-e['ci95'][0])],[100*(e['ci95'][1]-e['mean'])]],fmt='D',color=color,capsize=3,label=model if i==0 else None)
    ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(3),yticklabels=list(result['datasets']),xlabel='Change in accuracy (percentage points)',title=condition);ax.legend(ncol=3)
plt.show()''')
 md('''## Author-reference evidence · a separate recorded run
The following is the author's measured snapshot, not your kernel state. Compare protocol and arithmetic before interpreting numerical differences. Do not treat three paired seed intervals as uncertainty about future datasets.''')
 section('results');figure('effects','Author-reference local effects: per-seed differences and conditional 95% t intervals on a shared test set.')
 figure('ranks','Author-reference original-condition ranks: three tasks; no Nemenyi pair exceeds CD. Nonsignificance does not establish equivalence.')
 code('''# CHECK — recompute accuracy from predictions; verify the paired baseline.
for dataset,row in result['datasets'].items():
    for condition,runs in row['runs'].items():
        for model,rr in runs.items():
            for run in rr:
                assert np.isclose(run['accuracy'],accuracy_score(row['test_y'],np.array(run['probability'])>=.5))
    for model in MODELS:
        fresh=paired_effect([a['accuracy'] for a in row['runs']['smoothed'][model]],[a['accuracy'] for a in row['runs']['top5'][model]])
        assert np.isclose(fresh['mean'],row['effects']['smoothed'][model]['mean'])
print('CHECK: saved prediction arithmetic and smoothing baseline pass.')''')
 md('''## EXIT · Your interpretation is the deliverable
Complete in your own words, then paste this text and the printed report to the teacher:

1. **Smoothness:** changed labels / training rows; paired effect for one model; why evaluation labels stay raw.
2. **Rotation:** explain X′Rᵀ=X and W′=WR; why Adam training can still change the score.
3. **Noise:** report one exception or uncertainty; explain population independence versus accidental sample patterns.
4. **Claim limit:** identify the experimental unit, the correct smoothing baseline, and two missing paper-protocol components.
5. **Mission:** one control you will require before treating a future relational-model gain as evidence for the thesis.

YOUR ANSWER: ...''')
 provided(['code_fingerprint'],'stable executable identity; excludes interpreter reference bookkeeping',(HERE/'_live_identity_l051.py').read_text())
 code('''# EXIT TICKET — implementation identity and actual measured effects.
tracked=['smooth_targets','rotate_splits','add_noise_features','paired_effect','fit_arm','prepare_task','run_experiment','rank_summary','reglu']
live_identity={n:code_fingerprint(globals()[n]) for n in tracked}
for cls in [CheckpointMLP,CheckpointFT,CheckpointAttention,CheckpointBlock]:
    for method in ['__init__','forward']:
        live_identity[cls.__name__+'.'+method]=code_fingerprint(getattr(cls,method))
ticket=dict(lesson=51,protocol=result['protocol'],verdict='INCOMPARABLE',identity=live_identity,
            effects={n:r['effects'] for n,r in result['datasets'].items()},statistics=result['statistics'])
(LABS/'data/cache/l051-exit.json').write_text(json.dumps(ticket,indent=2))
(LABS/'data/cache/l051-teacher-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(ticket,indent=2))''')
 section('scale')
 md('''### Required larger attempt · explicit gate
The same experiment is inlined above. This operator checkpoints completed datasets, validates the implementation/data/configuration/environment identity and prints an honest conclusion ledger. Attach a GPU for a larger run, set `RUN_PAPER_REPRO=True`, and use a fresh output directory after editing code. The `paper` preset means a larger local resource budget, not source-protocol equivalence. Completing this cell cannot by itself reproduce the missing search curves.

The gate re-hashes the current notebook code when run, rather than relying on the earlier EXIT hash.''')
 provided(['reproduce'],'visible resume/operator contract',(HERE/'_paper_repro_l051.py').read_text())
 code('''# PROVIDED — Colab/local gate. Your current live runner is passed explicitly.
RUN_PAPER_REPRO=False
PAPER_PRESET='closer'
if RUN_PAPER_REPRO:
    live_identity={n:code_fingerprint(globals()[n]) for n in tracked+['load_task','reproduce','code_fingerprint']}
    for cls in [CheckpointMLP,CheckpointFT,CheckpointAttention,CheckpointBlock]:
        for method in ['__init__','forward']:
            live_identity[cls.__name__+'.'+method]=code_fingerprint(getattr(cls,method))
    live_identity['settings']=dict(DATASETS=DATASETS,PRESETS=PRESETS,CONDITIONS=CONDITIONS,MODELS=MODELS)
    # The inlined operator resolves its root from this temporary file identity.
    previous_file=__file__;__file__=str(LABS/'_paper_repro_l051.py')
    try:
        larger=reproduce(PAPER_PRESET,out=LABS/'data/cache/l051-student-closer',
            device='cuda' if torch.cuda.is_available() else 'cpu',runner=run_experiment,identity=live_identity)
    finally:__file__=previous_file
else:print('Scale-up gate OFF. Paper search curves remain NOT_RUN; author closer evidence is separately labeled.')''')
 return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'lesson':51})
if __name__=='__main__':
 for solution in [False,True]:
    path=HERE/('solutions' if solution else '')/f'{SLUG}.ipynb';path.parent.mkdir(exist_ok=True);nbf.write(build(solution),path);print(path)

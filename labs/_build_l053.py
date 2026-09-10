"""Build student + solution notebooks with visible model/training code and portable figures."""
from _lesson_depth import enrich_notebook
import ast,base64,os
from pathlib import Path
from urllib.parse import urlsplit
import nbformat as nbf
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_l049 import mdtext
ROOT=Path(__file__).resolve().parent;SLUG='0053-realmlp-strong-defaults'
SOURCE=(ROOT/'relkit/realmlp.py').read_text();EXPERIMENT=(ROOT/'relkit/realmlp_experiment.py').read_text()
ABLATION=(ROOT/'_ablation_l053.py').read_text()
LESSON=BeautifulSoup((ROOT.parent/'lessons'/f'{SLUG}.html').read_text(),'html.parser')
def extract(source,names):
    tree=ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)

def build(solution=False,write=True):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
    def figure(name,caption):md('!['+caption+'](data:image/png;base64,'+base64.b64encode((ROOT/'figures/l053'/f'{name}.png').read_bytes()).decode()+')\n\n'+caption+' [Open full-size figure](figures/l053/'+name+'.png). On a small screen, open the full-size figure to inspect labels and seed points.')
    def section(name):
        node=BeautifulSoup(str(LESSON.find('section',id=name)),'html.parser').find('section')
        for el in node.find_all(['figure','script']):el.decompose()
        for el in node.find_all('div',id=True):el.decompose()
        for link in node.find_all('a',href=True):
            parts=urlsplit(link['href'])
            if not parts.scheme and parts.path:
                link['href']=os.path.relpath((ROOT.parent/'lessons'/parts.path).resolve(),ROOT)+('#'+parts.fragment if parts.fragment else '')
        md(mdtext(node))
    def task(name,signature,prompt):
        md(prompt)
        code('# TODO — '+name+'\n'+(extract(SOURCE,{name}) if solution else signature+'\n    raise NotImplementedError("Implement this live function")'))
    md('''# Lab 053 · RealMLP & strong defaults

[Lesson](../lessons/0053-realmlp-strong-defaults.html) · [Reference](../reference/realmlp-strong-defaults.html)

**Skill:** implement numeric RealMLP-TD-S and audit a fixed recipe versus tuned XGBoost. **Scope:** complete numeric TD-S forward path and training recipe, binary classification and regression; not full TD, categories or the paper benchmark. Four TODO functions feed the live training code. PROVIDED = read/run; CHECK = immediate diagnostic feedback; EXIT = explain your result.

**Reproducibility contract:** Tier A real California Housing, House 16H and Higgs Small data from the TabR release, not RealMLP benchmark splits. Preserve released boundaries; label-blind row caps 1200/600/600, selection seeds 53/54/55; model seeds 0/1/2. Train-only numeric statistics and target standardization. Learning recipe: width 64, three hidden layers, 64 epochs, batch 256, all four schedule cycles; six XGB candidates, 150 trees each, patience 20. **INCOMPARABLE** to the paper benchmark. See [contract](l053-reproduction.md).

The historical comparison took about 30 CPU seconds after setup; the added clipping intervention runs nine more neural fits and may take another minute. Downloading dependencies/data takes longer. Implementation and interpretation deserve a separate session. Static figures are labeled author-reference snapshots; your live results appear only when you run the experiment.

**Recall first:** Explain why three random model seeds do not give three independent datasets. Then predict whether selecting a lower validation error guarantees a lower test error.''')
    for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
    code('''# PROVIDED — runtime and data setup
import os,sys,math,copy,hashlib,json,time,importlib.metadata
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import rankdata,friedmanchisquare,studentized_range
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier,XGBRegressor
torch.set_num_threads(1)
for candidate in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (candidate/'relkit').is_dir():
        LABS=candidate.resolve();os.chdir(LABS);sys.path.insert(0,str(LABS));break
else:raise RuntimeError('Use the Colab bootstrap or run inside this repository')
from _fetch_l052 import fetch
from relkit.tabr_experiment import seed_interval
fetch()
print({p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','xgboost']})
print('Data manifest SHA256',hashlib.sha256((LABS/'_data_l052.json').read_bytes()).hexdigest())''')
    section('defaults');figure('boundaries','Two evaluation levels: meta-train split test errors can guide recipe development; on separate meta-test datasets, freeze the recipe, fit fresh weights and select epochs.')
    code('''# CHECK — reconstruct the paper score on the synthetic example, not lab errors
example_a=np.array([.01,.41]);example_b=np.array([.04,.36])
score_a=np.exp(np.log(example_a+.01).mean())
score_b=np.exp(np.log(example_b+.01).mean())
assert example_a.mean()>example_b.mean() and score_a<score_b
np.testing.assert_allclose([score_a,score_b],[np.sqrt(.02*.42),np.sqrt(.05*.37)])
print('Synthetic arithmetic means:',example_a.mean(),example_b.mean())
print('Synthetic paper scores (no shift subtraction):',score_a,score_b)''')
    md('''### Retrieval check
You change a default after examining dataset E's test results, then make a fresh row split on E. Is E now a new meta-test dataset? Write an answer before opening the feedback.

<details><summary>Compare your reasoning</summary>No. You refreshed a row split, but E already influenced the recipe. Meta-test status concerns dataset-level development history.</details>''')
    section('architecture');figure('architecture','Model architecture: numeric TD-S, no pretraining or retrieval; published width 256, learning width 64.')
    section('preprocessing');figure('preprocessing','Synthetic smooth clipping trace: z=2 maps to 1.664, while the function approaches ±3 in the tails.')
    task('robust_parameters','def robust_parameters(x):','''### TODO 1 · Fit robust statistics
**Goal:** return two vectors, training-column medians and multiplicative scales. **Why:** new-task preprocessing must be defined even for sparse or constant columns. Input is a finite 2D NumPy array. Implement all three cases described above without dividing by zero. Do not use validation rows.''')
    code('''# CHECK — zero-IQR nonconstant, ordinary IQR, constant, and scale equivariance
x=np.array([[0.,1.,7.],[0.,2.,7.],[0.,3.,7.],[0.,4.,7.],[10.,5.,7.]])
m,s=robust_parameters(x)
np.testing.assert_allclose(m,[0,3,7]);np.testing.assert_allclose(s,[.2,.5,0])
m2,s2=robust_parameters(10*x+17)
np.testing.assert_allclose(m2,10*m+17);np.testing.assert_allclose(s2,s/10)
assert np.isfinite(s).all()
print('PASS: IQR fallback, constants and affine unit changes')''')
    task('smooth_clip','def smooth_clip(z):','''### TODO 2 · Compress the tails
**Goal:** implement the smooth transform elementwise for a NumPy array. **Why:** bound extreme inputs while retaining finite-value ordering. The recap defines the function; your code must handle positive, negative and zero inputs without changing array shape.''')
    code('''# CHECK — independent values, odd symmetry, order and strict finite bounds
z=np.array([-30.,-3.,0.,3.,30.]);v=smooth_clip(z)
np.testing.assert_allclose(v[1:4],[-3/np.sqrt(2),0,3/np.sqrt(2)])
np.testing.assert_allclose(v,-smooth_clip(-z));assert (np.diff(v)>0).all()
assert (np.abs(v)<3).all() and v.shape==z.shape
print('PASS: monotonic compression, not hard clipping')''')
    code('# PROVIDED — train-fitted transform wrapper\n'+extract(SOURCE,{'RobustSmooth'}))
    section('parameterization')
    task('ntp_linear','def ntp_linear(x, weight, bias):','''### TODO 3 · Width-scaled linear computation
**Goal:** return the NTP preactivation with W stored as [input,output]. **Why:** placing the scaling factor on the bias changes the function. Use torch operations so gradients reach inputs, weights and biases; support arbitrary batch size and widths.''')
    code('''# CHECK — two output coordinates; bias remains outside width scaling
x=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
w=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],requires_grad=True)
b=torch.tensor([[2.,-1.]],requires_grad=True)
out=ntp_linear(x,w,b);torch.testing.assert_close(out,torch.tensor([[4.,2.]]))
out.sum().backward();torch.testing.assert_close(b.grad,torch.ones_like(b))
assert x.grad.abs().sum()>0 and w.grad.abs().sum()>0
print('PASS: NTP arithmetic, bias location and gradients')''')
    code('# PROVIDED — full visible numeric model (Table A.1, Appendix A.2)\n'+extract(SOURCE,{'NTPLinear','RealMLPS'}))
    code('''# CHECK — zero output head; initially blocks upstream gradients
torch.manual_seed(53);model=RealMLPS(4,width=16,regression=True)
pred=model(torch.randn(9,4));assert torch.count_nonzero(pred)==0
pred.sum().backward();assert model.layers[-1].weight.grad.abs().sum()>0
assert model.scale.grad.abs().sum()==0
groups=model.parameter_groups();assert [g['factor'] for g in groups]==[6.,1.,.1]
params=[id(p) for g in groups for p in g['params']]
assert len(params)==len(set(params))==len(list(model.parameters()))
print('PASS: zero head and disjoint complete optimizer groups')''')
    md('''### Trace the first three backward passes
**Predict:** the head gradient is nonzero, but the schedule starts at zero. Do the hidden weights move on the next backward pass? Keep the distinction between computing a gradient, updating Adam's moments, and moving a parameter. The following CHECK uses the live model; two positive-rate updates are deliberately shown after the initial zero-rate step. This is a synthetic optimizer probe, not a trained dataset result.''')
    code('''# CHECK — zero rate, then moving head, then an upstream gradient
torch.manual_seed(530);probe=RealMLPS(4,width=16,regression=True)
probe_x=torch.randn(9,4);probe_y=torch.ones(9,1)
probe_opt=torch.optim.Adam(probe.parameter_groups(),betas=(.9,.95),eps=1e-8)
trace=[]
for label,base in [('first: zero rate',0.),('second: head moves',.01),('third: hidden route opens',.01)]:
    probe_opt.zero_grad();((probe(probe_x)-probe_y)**2).mean().backward()
    upstream=float(probe.scale.grad.abs().sum())
    head_grad=float(probe.layers[-1].weight.grad.abs().sum())
    before=[p.detach().clone() for p in probe.parameters()]
    for group in probe_opt.param_groups:group['lr']=base*group['factor']
    probe_opt.step()
    moved=max(float((p-old).abs().max().detach()) for p,old in zip(probe.parameters(),before))
    trace.append(dict(pass_name=label,head_gradient=head_grad,scale_gradient=upstream,max_parameter_change=moved))
assert trace[0]['head_gradient']>0 and trace[0]['max_parameter_change']==0
assert trace[1]['scale_gradient']==0 and trace[1]['max_parameter_change']>0
assert trace[2]['scale_gradient']>0
assert probe_opt.state[probe.layers[-1].weight]['step']==3
display(pd.DataFrame(trace))''')
    section('training');figure('schedule','Four-cycle schedule; valleys at 0, 1/15, 3/15, 7/15 and 1. Parameter-group rates differ even at the same training step.')
    task('coslog4','def coslog4(t):','''### TODO 4 · Reconstruct the schedule
**Goal:** return the schedule multiplier for normalized optimizer-step progress t. **Why:** an ordinary cosine or equally spaced restart schedule is a different experiment. Use the paper expression above, not a lookup table; the CHECK also probes mid-cycle peaks.''')
    code('''# CHECK — analytical valleys and peaks; non-grid argument
for j in range(5):assert abs(coslog4((2**j-1)/15))<1e-12
for j in range(4):assert abs(coslog4((2**(j+.5)-1)/15)-1)<1e-12
assert 0<=coslog4(.371)<=1
print('PASS: four logarithmically spaced cycles')''')
    code('# PROVIDED — validation tie policy\n'+extract(SOURCE,{'last_best'})+'\nassert last_best([.4,.3,.35,.3])==3')
    section('comparison')
    code('# PROVIDED — data and metric helpers (statistics fit on training only)\n'+extract(EXPERIMENT,{'load_task','error'}))
    code('# PROVIDED — visible neural training loop, calls your model and schedule\n'+extract(EXPERIMENT,{'fit_neural'}))
    md('''### Read the selection boundary before training
`fit_neural` reads validation errors for checkpoint selection and accesses test predictions after restoring that checkpoint. `fit_trees` selects one of six candidates using validation, then scores only the fixed and selected candidates on test. Predict which method will lead on each task, and whether tree tuning will improve every test score. No CHECK requires a particular winner.''')
    code('# PROVIDED — declared search and visible tree selection\nSEARCH = [dict(max_depth=d, learning_rate=lr) for d in (3,6) for lr in (.03,.1,.2)]\n\n'+extract(EXPERIMENT,{'fit_trees','run_suite'}))
    code('''# PROVIDED — live local experiment (about 30 CPU seconds on the author machine)
live_results=run_suite(RealMLPS,prep_class=RobustSmooth)
rows=[]
for name,data in live_results['results'].items():
    for arm,s in data['summary'].items():
        rows.append(dict(dataset=name,method=arm,metric=data['metric'],mean=s['mean'],sd=s['sd'],ci95=s['ci95']))
display(pd.DataFrame(rows))
display(pd.DataFrame([live_results['ranks']['means']]).T.rename(columns={0:'mean rank'}))
print('Friedman p',live_results['ranks']['friedman_p'],'CD',live_results['ranks']['nemenyi_cd'])''')
    code('''# CHECK — reconstruct saved errors and selection without trusting summary fields
for name,result in live_results['results'].items():
    data=load_task(name)
    for arm,runs in result['runs'].items():
        for run in runs:
            measured=error(np.array(run['prediction']),data['y']['test'],data['regression'],data['target_std'])
            assert abs(measured-run['error'])<max(1e-6,abs(measured)*1e-6)
            if arm=='RealMLP-S':assert run['best_epoch']==last_best(run['history'])+1
            elif arm=='XGB-tuned':assert run['selected']==int(np.argmin([c['validation_error'] for c in run['search']]))
print('PASS: all 27 errors and validation selections reconstructed')''')
    code('''# PROVIDED — plot your current kernel results; not the author snapshot
fig,axes=plt.subplots(1,3,figsize=(11,4))
for ax,(name,data) in zip(axes,live_results['results'].items()):
    for i,(arm,runs) in enumerate(data['runs'].items()):
        values=[r['error'] for r in runs];s=seed_interval(values)
        ax.scatter([i-.08,i,i+.08],values,label=arm)
        ax.errorbar(i,s['mean'],yerr=[[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]],color='black',capsize=3)
    ax.set_title(name);ax.set_ylabel(data['metric']+' ↓');ax.set_xticks([0,1,2],['TD-S','XGB-fixed','XGB-tuned'],rotation=20)
fig.suptitle('Your live runs: seeds and conditional 95% intervals');fig.tight_layout();plt.show()''')
    section('evidence');figure('results','Author-reference local measurement, separate from your current kernel output: errors with seed dots and conditional 95% t intervals.');figure('ranks','Author-reference ranks across three tasks. A nonsignificant low-power test is not evidence of equivalence.')
    section('ablation');figure('ablation','Author-reference paired intervention: remove only smooth clipping. Positive differences mean worse error; every conditional interval includes zero.')
    md('''### Live intervention · remove just smooth clipping
**Goal:** test the same distinctive preprocessing choice using your live TODO functions. **Why:** a neural-versus-tree score cannot isolate clipping's contribution. **Held fixed:** fitted robust statistics, selected rows, model seeds, widths, losses, rates, schedule and validation rule. **Varied:** the transform after robust scaling. **Measured:** paired test-error changes, one per seed. **Hint:** pair by seed identity, not by list position or separate confidence-interval endpoints. Predict the signs before running. This is a local diagnostic and has no required winner.

The subclass below inherits your live `RobustSmooth.fit`, so the ablated arm still uses your robust-statistics TODO. Both arms use your NTP and schedule TODOs. Only the baseline uses your smooth-clip TODO. We intentionally reuse your already completed baseline fits; retraining them again would add cost without changing this comparison.''')
    code('# PROVIDED — one-component intervention and seed-paired uncertainty\n'+extract(ABLATION,{'RobustOnly','paired_effect'}))
    code('''# PROVIDED — live intervention, same local recipe; nine extra neural fits
ablation_results=run_suite(RealMLPS,prep_class=RobustOnly,neural_only=True)
paired_results={}
for name,baseline in live_results['results'].items():
    changed=ablation_results['results'][name]
    assert baseline['selection']==changed['selection'] and baseline['hashes']==changed['hashes']
    paired_results[name]=dict(metric=baseline['metric'],**paired_effect(
        baseline['runs']['RealMLP-S'],changed['runs']['RealMLP-S']))
display(pd.DataFrame(paired_results).T[['metric','differences','mean','sd','ci95']])
# CHECK — pairing survives an order change; stored predictions give the same errors
for name,changed in ablation_results['results'].items():
    data=load_task(name);baseline=live_results['results'][name]['runs']['RealMLP-S']
    reverse=paired_effect(baseline,list(reversed(changed['runs']['RealMLP-S'])))
    np.testing.assert_allclose(reverse['mean'],paired_results[name]['mean'])
    for run in changed['runs']['RealMLP-S']:
        np.testing.assert_allclose(error(np.array(run['prediction']),data['y']['test'],
            data['regression'],data['target_std']),run['error'],rtol=1e-6)
print('PASS: same rows, seed pairing and nine reconstructed intervention scores')''')
    md('''### EXIT TICKET · complete in your own words
Report one selected neural epoch and one selected tree configuration. Explain why the test set did not choose either. Compare fixed and tuned XGB on one task, including units and seed uncertainty. State the number of independent datasets and interpret Friedman without equating non-rejection with equality. Give the paper verdict and two protocol deviations. Explain how a default can embody extensive tuning. Interpret one paired clipping result and propose a next experiment that addresses its uncertainty. Explain why Table B.2's relative SGM changes are not percentage-point changes in our Higgs error.

Paste your output and explanation to the teacher for feedback; passing numerical CHECKs alone does not grade the explanation.''')
    explanation = ('Validation selected the epoch and tree configuration before test scoring. The three datasets, not the three model seeds, are the cross-task units. Non-rejection in Friedman does not show equal methods. Clipping removal has uncertain paired effects here; repeat predeclared splits before drawing a stable conclusion. The paper uses ten random splits and benchmark SGM, whereas this run uses one capped split and model-seed intervals. Default recipes can be developed on other datasets. Verdict: INCOMPARABLE.' if solution else '')
    code('''# EXIT — replace the blank explanation before submitting
interpretation = '''+repr(explanation)+'''  # TODO: write your own conclusion and protocol limits
print('Paper verdict:',live_results['verdict'])
print('Dataset count:',len(live_results['results']))
print('Mean ranks:',live_results['ranks']['means'])
assert interpretation.strip(), 'Complete the written EXIT explanation before saving your submission'
from _paper_repro_l053 import identity
exit_path=Path('data/cache/l053-student/exit.json');exit_path.parent.mkdir(parents=True,exist_ok=True)
exit_artifact=dict(lesson=53,paper_verdict='INCOMPARABLE',interpretation=interpretation,
    comparison=live_results,intervention=ablation_results,paired=paired_results,
    live_code_identity=identity(RealMLPS,run_suite,RobustSmooth,RobustOnly,paired_effect))
exit_path.write_text(json.dumps(exit_artifact,indent=2))
print('Saved reproducible EXIT:',exit_path.resolve())
print('Your explanation:',interpretation)''')
    section('scaleup')
    closer=ROOT/'figures/l053/closer.png'
    if closer.exists():figure('closer','Author-reference larger CPU run: full published width and epochs, 6000 training rows, three seeds. Still a different dataset protocol from the paper.')
    code('''# PROVIDED — required NEXT STEP, live code; gated OFF by default
RUN_PAPER_REPRO = False
PAPER_PRESET = 'closer'  # smoke / closer / paper (model settings, not benchmark parity)
if RUN_PAPER_REPRO:
    from _paper_repro_l053 import reproduce
    scaleup=reproduce(PAPER_PRESET,out=f'data/cache/l053-notebook-{PAPER_PRESET}',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_class=RealMLPS,runner=run_suite,prep_class=RobustSmooth)
    display(pd.DataFrame([scaleup['summary']]))
    print(scaleup['ledger'])
else:
    print('Scale-up for YOUR live code: NOT_RUN. Author larger-run evidence is separate.')''')
    md('''### Return tomorrow
Without reopening the formula, explain the two evaluation levels and reconstruct the zero-IQR fallback. Then name the difference between the TD and TD-S variants. Ask the teacher about any CHECK or derivation you cannot explain.''')
    nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3'))),53)
    if write:
        path=ROOT/('solutions' if solution else '')/f'{SLUG}.ipynb';path.parent.mkdir(exist_ok=True);nbf.write(nb,path);print(path)
    return nb

if __name__=='__main__':build(False);build(True)

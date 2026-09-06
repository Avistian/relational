"""Build the standalone TabR lesson/lab from canonical code, prose and figures."""
import ast,base64,json,os
from pathlib import Path
from urllib.parse import urlsplit
import nbformat as nbf
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_l047 import extract
from _build_l049 import mdtext
HERE=Path(__file__).resolve().parent;SLUG='0052-tabr-retrieval'
SOURCE=(HERE/'relkit/tabr.py').read_text();EXPERIMENT=(HERE/'relkit/tabr_experiment.py').read_text()
LESSON=BeautifulSoup((HERE.parent/'lessons'/f'{SLUG}.html').read_text(),'html.parser')

def build(solution=False,write=True):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 def figure(name,caption):md('!['+caption+'](data:image/png;base64,'+base64.b64encode((HERE/'figures/l052'/f'{name}.png').read_bytes()).decode()+')\n\n'+caption)
 def section(name):
  node=BeautifulSoup(str(LESSON.find('section',id=name)),'html.parser').find('section')
  for el in node.find_all(['figure','script']):el.decompose()
  for link in node.find_all('a',href=True):
   parts=urlsplit(link['href'])
   if not parts.scheme and parts.path:
    target=(HERE.parent/'lessons'/parts.path).resolve();link['href']=os.path.relpath(target,HERE)+('#'+parts.fragment if parts.fragment else '')
  md(mdtext(node))
 def task(name,stub):code('# TODO — teacher implementation\n'+extract(SOURCE,{name}) if solution else '# TODO — implement the live function\n'+stub)
 md('''# Lab 052 · TabR: learn what to retrieve

[Lesson](../lessons/0052-tabr-retrieval.html) · [Reference](../reference/tabr-retrieval.html)

**Skill:** implement and audit kNN attention. **Scope:** complete numeric TabR-S (linear encoder, one predictor block), Equation 5; binary classification and regression. Full TabR feature embeddings, categorical inputs, context freeze and multiclass output are not implemented.

Four TODO functions feed the actual training experiment. PROVIDED means read/run; CHECK gives immediate feedback; EXIT asks you to explain both the mechanism and its evidence limits. Author-reference figures are already measured snapshots, separate from your live outputs.

**Reproducibility contract:** Tier A, authors' California Housing, House 16H and Higgs Small arrays and original splits. Local row caps 1200/600/600; label-blind subsample seeds 52/53/54, model seeds 0/1/2; train-only normal quantiles and regression target scaling; 25-epoch neural cap, 150-tree cap. No quantile jitter or tuning search. Local versus paper verdict: **INCOMPARABLE**. Environment versions and file hashes are printed. The bootstrap downloads dependencies; a first data fetch retrieves about 14 MB in verified byte ranges from the authors' archive.

**Recall first:** Why are three model seeds not three independent datasets? Write one sentence before proceeding. Estimated compute is about one CPU minute after setup; allow much longer for implementation and explanation.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 code('''# PROVIDED — environment and data provisioning
import os,sys,math,hashlib,json,copy,time,platform,types,importlib.metadata
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import numpy as np
import torch
from torch import nn
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
torch.set_num_threads(1)
for candidate in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (candidate/'relkit').is_dir():
        LABS=candidate.resolve();os.chdir(LABS);sys.path.insert(0,str(LABS));break
else:raise RuntimeError('Run inside the repository or use the Colab bootstrap')
from _fetch_l052 import fetch
fetch()
print({p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','xgboost']})
print('Dataset manifest SHA256:',hashlib.sha256((LABS/'_data_l052.json').read_bytes()).hexdigest())''')
 section('idea');section('architecture');figure('architecture','Complete numeric TabR-S. Read the shared encoder, corrected neighbor values, residual route and prediction head before implementing retrieval.')
 section('neighbors');figure('exclusion','Synthetic top-2 attention at query key 1. Excluding the identical row ID removes the direct target path; equal feature vectors with distinct IDs remain separate candidates.')
 md('''### TODO 1 · Legal identities
Implement a boolean matrix of shape [queries,candidates]. Entry (i,j) is true exactly when the query and candidate IDs differ. Support arbitrary candidate ordering. Do not compare feature vectors.''')
 task('eligible_mask','def eligible_mask(query_ids, candidate_ids):\n    raise NotImplementedError("Build the legal identity matrix")')
 code('''# CHECK — reordered IDs, repeated feature values irrelevant
assert eligible_mask(torch.tensor([9,3]),torch.tensor([3,7,9])).tolist()==[[True,True,False],[False,True,True]]
assert eligible_mask(torch.tensor([8]),torch.tensor([1,2,3])).all()
print('Identity boundary PASS')''')
 md('''### TODO 2 · Discrete legal neighbors
Compute squared distances between every query and candidate key. Inside a no-gradient region, make illegal distances ineligible and return the m nearest indices per query. Raise ValueError if m is invalid or any query has too few legal candidates. Gradients will be reconstructed on selected keys in the model.''')
 task('select_neighbors','def select_neighbors(keys, candidate_keys, m, allowed):\n    raise NotImplementedError("Select m legal nearest indices")')
 code('''# CHECK — more than the illustrated example, and no silent illegal fallback
q=torch.tensor([[0.,0.],[2.,0.]])
c=torch.tensor([[2.,0.],[0.,0.],[0.,0.]])
mask=eligible_mask(torch.tensor([9,3]),torch.tensor([3,7,9]))
assert select_neighbors(q,c,1,mask).tolist()==[[1],[1]]
torch.manual_seed(520)
q2=torch.randn(4,5);c2=torch.randn(13,5);legal=torch.ones(4,13,dtype=torch.bool);legal[:,2]=False
got=select_neighbors(q2,c2,4,legal)
brute=(q2[:,None]-c2[None]).square().sum(-1).masked_fill(~legal,torch.inf).argsort(dim=1)[:,:4]
assert torch.equal(got,brute)
try:select_neighbors(q,c,3,mask)
except ValueError:pass
else:raise AssertionError('Illegal context must fail')
print('Neighbor selection PASS')''')
 section('values');figure('values','Synthetic scalar illustration: changing β changes per-neighbor values; equal weights and opposite key differences make the corrections cancel in the sum.')
 md('''### TODO 3 · Corrected vector values
Return one d-dimensional value per selected neighbor. Add the supplied label embeddings to the supplied correction module evaluated on the correctly directed key difference. Broadcasting must work for B queries and m neighbors.''')
 task('context_value','def context_value(keys, neighbor_keys, label_embeddings, correction):\n    raise NotImplementedError("Construct Eq. 5 values")')
 code('''# CHECK — direction, shape, and gradient access
q=torch.tensor([[1.,2.]],requires_grad=True)
n=torch.tensor([[[0.,0.],[2.,3.]]],requires_grad=True)
e=torch.tensor([[[0.,1.],[1.,0.]]])
v=context_value(q,n,e,nn.Identity())
torch.testing.assert_close(v,torch.tensor([[[1.,3.],[0.,-1.]]]))
v.sum().backward();assert q.grad.abs().sum()>0 and n.grad.abs().sum()>0
print('Directed values and gradients PASS')''')
 md('''### TODO 4 · Weights and aggregation
Compute negative squared distance over the key dimension, normalize over the neighbor dimension, apply the supplied dropout module, then return the weighted sum of vector values. Do not insert dimension scaling or renormalize after dropout. The checks include an asymmetric neighborhood where using the wrong softmax axis is detectable.''')
 task('aggregate_context','def aggregate_context(keys, neighbor_keys, values, dropout):\n    raise NotImplementedError("Weight and sum the context")')
 code('''# CHECK — asymmetric weights, translation invariance, dropout semantics
q=torch.tensor([[0.,0.]]);n=torch.tensor([[[0.,0.],[1.,0.]]]);v=torch.tensor([[[2.,-1.],[0.,3.]]])
w=torch.tensor([1.,math.exp(-1)]);w=w/w.sum()
expected=(v*w[None,:,None]).sum(1)
torch.testing.assert_close(aggregate_context(q,n,v,nn.Identity()),expected)
torch.testing.assert_close(aggregate_context(q+3,n+3,v,nn.Identity()),expected)
class Twice(nn.Module):
    def forward(self,x):return 2*x
torch.testing.assert_close(aggregate_context(q,n,v,Twice()),2*expected)
print('Aggregation PASS')''')
 md('''### PROVIDED · Complete model around your four functions
The linear input map, learned keys, label encoder, correction network, residual block and prediction head are all visible below. Calls resolve to your current function definitions; nothing imports a hidden replacement model.''')
 code(extract(SOURCE,{'TabRS'}))
 code('''# CHECK — live model: query batching and memory order do not change predictions
torch.manual_seed(52)
model=TabRS(3,d=8,m=3,dropout=0,context_dropout=0).eval()
x=torch.randn(9,3);y=torch.arange(9)%2;ids=torch.arange(9)
batch=model(x[:2],x,y,ids[:2],ids)
single=torch.cat([model(x[i:i+1],x,y,ids[i:i+1],ids) for i in range(2)])
torch.testing.assert_close(batch,single)
perm=torch.randperm(9)
torch.testing.assert_close(batch,model(x[:2],x[perm],y[perm],ids[:2],ids[perm]))
changed=y.clone();changed[0]=1-changed[0]
torch.testing.assert_close(batch[:1],model(x[:1],x,changed,ids[:1],ids))
print('Live model protocol PASS')''')
 section('legality');figure('availability','Synthetic availability check: at prediction day 4, only A has both an earlier event and an earlier known label.')
 md('''### PROVIDED · Source audit boundary
The author executed the pinned released Model class with copied weights against this implementation. Four cases cover regression/classification and training/inference candidate paths. Maximum output error was 1.20e-7 and maximum input-gradient error 5.97e-8. Dropout was disabled, and a PyTorch exact-search adapter replaced Faiss. This establishes the checked neural path, not full training, preprocessing or Faiss runtime parity. See `_source_check_l052.py` and `_source_check_l052_results.json` for the reproducible audit.''')
 section('evidence');figure('scores','Author-reference measurements. Dots are three model seeds; error bars are sample SD. These are fixed-split local results, not paper metrics.');figure('ranks','Author-reference mean ranks with Nemenyi CD over three datasets. Nonsignificance does not establish equivalence.')
 md('''### PROVIDED · Visible preprocessing, training and evaluation
Read `load_task`: transforms fit only on the training subset. Read `fit_neural`: training supplies row IDs for self-exclusion; validation/test queries see only the training memory. Validation chooses the checkpoint. `run_suite` receives your live TabRS class and calls it in each fit.

**Predict before running:** does the label-shuffling intervention estimate what a model trained without labels would achieve? Explain why not, then run.''')
 code(EXPERIMENT)
 code('''# PROVIDED — measured live run, about one CPU minute in the author environment
live_results=run_suite(TabRS,data_root=LABS/'data/cache/l052')
rows=[]
for name,task in live_results['results'].items():
    for arm,summary in task['summary'].items():
        rows.append(dict(dataset=name,metric=task['metric'],model=arm,**summary))
display(pd.DataFrame(rows))
print('Rank statistics:',live_results['stats'])
for name,task in live_results['results'].items():
    clean=[r['score'] for r in task['runs']['TabR-S']]
    changed=[r['shuffled_label_score'] for r in task['runs']['TabR-S']]
    print(name,'clean:',seed_interval(clean),'shuffled:',seed_interval(changed))''')
 code('''# CHECK — saved predictions must reconstruct reported metrics
for name,task in live_results['results'].items():
    for arm,runs in task['runs'].items():
        for run in runs:
            reconstructed=score_predictions(np.array(run['prediction']),np.array(task['test_target']),task['metric']=='RMSE',task['target_std'])
            assert abs(reconstructed-run['score'])<1e-7*max(1,abs(reconstructed))
print('Prediction arithmetic PASS')''')
 code('''# PROVIDED — live results plot; shared axes only within the same task
fig,axes=plt.subplots(1,3,figsize=(11,3.5))
for ax,(name,task) in zip(axes,live_results['results'].items()):
    for i,(arm,s) in enumerate(task['summary'].items()):
        ax.errorbar(i,s['mean'],yerr=s['sd'],fmt='o',capsize=4)
    ax.set_xticks(range(3),['MLP','TabR-S','XGB']);ax.set_title(name);ax.set_ylabel(task['metric'])
fig.suptitle('Your run: mean ± sample SD over model seeds');fig.tight_layout();plt.show()''')
 md('''### EXIT TICKET
Paste the report below and write: (1) one query’s identity→score→weight→value→output trace; (2) why your own target never enters your context; (3) one historical label-availability failure; (4) what the label permutation measures; (5) why these scores do not reproduce the paper’s broad comparison. Ask the teacher about any surprising score or shape. No glossary mastery is inferred from running cells.''')
 code('''# EXIT — attach your written explanation to this measured evidence
exit_report={'lesson':52,'metrics':rows,'ranks':live_results['stats'],'paper_verdict':'INCOMPARABLE','scope':live_results['protocol']}
print(json.dumps(exit_report,indent=2))
(LABS/'data/cache/l052-exit.json').write_text(json.dumps(exit_report,indent=2));''')
 section('scale')
 md('''### NEXT STEP · Same live code, larger data
The operator below uses your live model, helper functions and training loop. It fingerprints executable code and checks the data/configuration/environment before reusing completed seeds. Run the smoke preset first. Then attach a GPU if desired and enable the closer run. This is a required reproduction path, not evidence that it has run in your current kernel.''')
 # Only harness utilities are imported; never overwrite the student's model/runner.
 scale=(HERE/'_paper_repro_l052.py').read_text();tree=ast.parse(scale)
 drops=[]
 for n in tree.body:
  if isinstance(n,ast.ImportFrom) and n.module in ['relkit.tabr','relkit.tabr_experiment']:drops.append((n.lineno,n.end_lineno))
  if isinstance(n,ast.If):drops.append((n.lineno,n.end_lineno))
 lines=scale.splitlines();scale='\n'.join(l for i,l in enumerate(lines,1) if not any(a<=i<=b for a,b in drops))
 code('# PROVIDED — visible scale-up loop; __file__ identifies its canonical operator\n__file__=str(LABS/"_paper_repro_l052.py")\n'+scale)
 code('''# PROVIDED — fast live-code smoke gate, then a cached resume
smoke_identity=live_identity(TabRS,run_suite)
tag=hashlib.sha256(json.dumps(smoke_identity,sort_keys=True).encode()).hexdigest()[:12]
smoke_out=LABS/'data/cache'/('l052-notebook-smoke-'+tag)
smoke=reproduce('smoke',smoke_out,'cpu',model_class=TabRS,runner=run_suite)
resumed=reproduce('smoke',smoke_out,'cpu',model_class=TabRS,runner=run_suite)
assert smoke['summary']==resumed['summary']
assert live_identity(TabRS,run_suite)==smoke_identity
print('Live-code gate and completed-seed resume PASS')''')
 code('''# PROVIDED — opt in after checking resources; do not leave Colab unattended
RUN_PAPER_REPRO=False
PAPER_PRESET='closer'
if RUN_PAPER_REPRO:
    repro=reproduce(PAPER_PRESET,LABS/'data/cache'/('l052-student-'+PAPER_PRESET),
        'cuda' if torch.cuda.is_available() else 'cpu',model_class=TabRS,runner=run_suite)
    print(repro['ledger'])
else:
    print('Current-kernel scale-up: NOT_RUN. See the labeled author evidence above.')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 if write:
  out=HERE/('solutions' if solution else '')/f'{SLUG}.ipynb';out.parent.mkdir(parents=True,exist_ok=True);nbf.write(nb,out);print(out)
 return nb

if __name__=='__main__':build();build(True)

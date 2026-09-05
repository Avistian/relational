"""Build the independently teachable L049 notebook from lesson prose and visible code."""
import base64
import json
from pathlib import Path
import nbformat as nbf
from bs4 import BeautifulSoup, NavigableString, Comment
from _build_l047 import extract
from _colab import bootstrap_cells

HERE=Path(__file__).resolve().parent
MODEL=(HERE/'relkit/claim_models.py').read_text()
HARNESS=(HERE/'_verify_l049.py').read_text()
REPRO=(HERE/'_paper_repro_l049.py').read_text()
LESSON=BeautifulSoup((HERE.parent/'lessons/0049-excelformer-trompt.html').read_text(),'html.parser')


def mdtext(node):
    if isinstance(node,Comment):return ''
    if isinstance(node,NavigableString):return str(node)
    if node.name in ('figure','script','nav','details'):return ''
    if node.name=='div' and node.get('id'):return ''
    body=''.join(mdtext(c) for c in node.children)
    if node.name=='a':return '['+body+']('+node.get('href','')+')'
    if node.name in ('strong','b'):return '**'+body+'**'
    if node.name in ('em','i'):return '*'+body+'*'
    if node.name=='code':return '`'+body+'`'
    if node.name in ('h2','h3'):return '\n\n'+('#'*int(node.name[1]))+' '+body+'\n\n'
    if node.name=='br':return '  \n'
    if node.name in ('p','div'):return '\n\n'+body+'\n\n'
    if node.name=='table':
        rows=[[c.get_text(' ',strip=True) for c in tr.find_all(['th','td'])] for tr in node.find_all('tr')]
        return '\n\n'+'\n'.join(['| '+' | '.join(r)+' |' for r in [rows[0],['---']*len(rows[0])]+rows[1:]])+'\n\n'
    return body


def build(solution=False):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
    def section(key):md(mdtext(LESSON.find('section',id=key)))
    def provided(names,why):code('# PROVIDED — '+why+'\n'+extract(MODEL,set(names)))
    def task(names,student):code('# TODO — teacher answer\n'+extract(MODEL,set(names)) if solution else student)
    def figure(name,caption):
        data=base64.b64encode((HERE/'figures/l049'/f'{name}.png').read_bytes()).decode()
        md(f'![{caption}](data:image/png;base64,{data})\n\n{caption}')
    md('''# Lab 049 · ExcelFormer & Trompt
## Make the claim fit the experiment.

[Lesson](../lessons/0049-excelformer-trompt.html) · [Audit reference](../reference/claim-audit.html)

**Your skill:** trace information routes, implement the crucial operations, then write a source-faithful claim audit. PROVIDED is readable implementation; TODO is your work; CHECK gives immediate feedback; EXIT is your evidence and explanation.

**Scope:** full numeric ExcelFormer prediction path and training loop, plus Trompt Eq.4–5 key parts. We do not train a full Trompt model or reproduce either large benchmark. Three author-released numeric tasks are Tier A. MovieLens is a separate real timestamp transfer demonstration. Synthetic matrices isolate mechanisms only.

**Route:** recall → masked attention → visible model → Feat-Mix → Trompt routing → paper protocol → three-task experiment → time-transfer audit → EXIT → larger Pima reproduction attempt.

The recorded full local experiment took about nine minutes of CPU computation; reading and implementing take longer. All figures are inline PNG data URLs, visible without execution or image downloads. Author-reference snapshots are labeled and are not your current kernel outputs. Local image validation does not certify the live Colab frontend.

**Before reading:** explain why L048’s exact forward check could coexist with an INCOMPARABLE MovieLens score. Write a sentence now; revisit it at EXIT.''')
    cells.extend(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']) for c in bootstrap_cells())
    code('''# PROVIDED — imports and thread limits before numerical work.
import os, sys, copy, math, json, hashlib, inspect
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
thread_limit = threadpool_limits(limits=1)
torch.set_num_threads(1)
# Kernels may start at repo root, labs/, or labs/solutions/.
for candidate in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (candidate/'relkit').is_dir():
        os.chdir(candidate);sys.path.insert(0,str(candidate));break
from relkit.claim_data import paper_data, temporal_data
from relkit.saint_experiment import summarize, environment
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss
from IPython.display import display
print(environment())''')
    section('claims');section('excel');figure('excel-architecture','Complete numeric ExcelFormer routing, including training-only mixing and the repeated block.')
    section('spa');figure('spa','Synthetic intervention: keep equal logits; change weak value 9→18; compare the same receivers with and without masking.')
    md('''### Task 1 — encode a permission rule
**Why:** a transposed mask can pass every shape check while implementing the opposite model. Inputs have shape [...,C,d_head]. Columns arrive strongest first. Compute scores, block weaker senders, normalize over senders, and return the weighted values plus the pre-dropout attention matrix. Keep dropout behavior provided below.''')
    task(['spa_attention'],'''# TODO — ExcelFormer Eq.1/2, released positional tie convention.
def spa_attention(q, k, v, dropout=0., training=False):
    scores = ____
    blocked = ____  # boolean [receiver, sender]
    weights = ____
    return F.dropout(weights, dropout, training) @ v, weights''')
    code('''# CHECK — values, direction, and forbidden gradient pathway.
q=torch.zeros(1,3,2,dtype=torch.double)
v=torch.tensor([[[2.,0.],[4.,0.],[9.,0.]]],dtype=torch.double,requires_grad=True)
z,a=spa_attention(q,q,v)
assert torch.allclose(z[0,:,0],torch.tensor([2.,3.,5.],dtype=torch.double)), 'Inspect sender/receiver orientation.'
assert torch.allclose(a.sum(-1),torch.ones(1,3,dtype=torch.double)), 'Normalize senders, not receivers.'
z[0,0,0].backward()
assert torch.equal(v.grad[0,:,0],torch.tensor([1.,0.,0.],dtype=torch.double)), 'Weak values must have zero path into strong attention output.'
print('CHECK 1 passed: outputs and blocked gradients agree.')''')
    md('''### Read the complete model
These PROVIDED chunks form the model you will train. The block calls **your** `spa_attention`. Tokenization is a featurewise gated map; attention is the only token-mixing operation inside each block. The head subsequently pools all columns, so it can still read a weak feature.''')
    provided(['GatedTokenizer'],'numeric feature embeddings — §3.2')
    provided(['SPABlock'],'attention residual followed by gated residual — released forward path')
    provided(['ExcelFormer'],'complete prediction path; no model package hidden behind this cell')
    code('''# CHECK — validate your visible model against the canonical implementation.
from relkit.claim_models import ExcelFormer as CanonicalExcel
from relkit.claim_models import spa_attention as canonical_attention

torch.manual_seed(49)
visible=ExcelFormer(5,d=16,heads=2,layers=2,dropout=0).double().eval()
reference=CanonicalExcel(5,d=16,heads=2,layers=2,dropout=0).double().eval()
reference.load_state_dict(visible.state_dict())
x=torch.randn(7,5,dtype=torch.double,requires_grad=True)
xref=x.detach().clone().requires_grad_()
a,b=visible(x),reference(xref)
assert torch.allclose(a,b,atol=1e-10,rtol=1e-10),'Your model differs from the equation-checked canonical path.'
a.sum().backward();b.sum().backward()
assert torch.allclose(x.grad,xref.grad,atol=1e-10,rtol=1e-10),'Input gradient mismatch.'
released=json.loads(Path('_reference_l049_results.json').read_text())
print('Your model matches the canonical path. Separate pinned released-code check:',released)
# The released-code artifact validates canonical code; this cell separately validates your code.''')
    section('augmentation');figure('mix','Synthetic donors 1 and 0: retaining feature importance 6 out of 10 gives target 0.600, versus count-based 0.333.')
    md('''### Task 2 — make the label follow the retained information
**Why:** swapping tensors correctly while mixing labels by the wrong share changes the learning objective. A boolean mask [B,C] chooses donor A. Handle zero total importance explicitly; the fallback is the retained-feature fraction.''')
    task(['feat_mix'],'''# TODO — ExcelFormer Eq.7/8, plus an explicit zero-information fallback.
def feat_mix(tokens, y, importance, mask, permutation):
    total = importance.sum()
    ratio = ____ if total > 0 else ____
    mixed = ____
    return mixed, ratio*y + (1-ratio)*y[permutation]''')
    code('''# CHECK — mixed values, targets, all-retained, and zero-information behavior.
tokens=torch.arange(12.).reshape(2,3,2);y=torch.tensor([1.,0.]);imp=torch.tensor([6.,3.,1.]);perm=torch.tensor([1,0])
mask=torch.tensor([[True,False,False],[True,False,False]])
mixed,target=feat_mix(tokens,y,imp,mask,perm)
assert torch.allclose(target,torch.tensor([.6,.4])), 'Weight labels by retained MI.'
assert torch.equal(mixed[:,0],tokens[:,0]) and torch.equal(mixed[:,1:],tokens[perm,1:]), 'Check donor direction.'
_,all_y=feat_mix(tokens,y,imp,torch.ones_like(mask),perm)
assert torch.equal(all_y,y)
_,zero_y=feat_mix(tokens,y,torch.zeros(3),mask,perm)
assert torch.allclose(zero_y,torch.tensor([1/3,2/3])), 'Zero total MI needs a finite declared fallback.'
print('CHECK 2 passed.')''')
    md('''### Read the learning loop
Validation log loss selects the checkpoint. Test labels are never used inside this loop. Feat-Mix runs only in training; prediction applies sigmoid to logits. Small initialized mixing weights are not frozen: optimizer updates can change them.''')
    provided(['train_excel'],'training, optional Feat-Mix, validation selection, checkpoint restore')
    provided(['predict_excel'],'batched, independent-row inference')
    section('trompt');figure('trompt-architecture','Full Trompt paper routing. The notebook implements Eq.4/5; other modules remain unimplemented and untrained here.')
    figure('prompt','Synthetic one-prompt example: fixed column vectors and values, query changes [0,0]→[2,0]; output changes 5.000→2.959.')
    md('''### Task 3 — preserve the right axes
**Why:** mixing up columns and prompts yields plausible shapes in symmetric examples. Use unequal P and C in the checks. Implement Eq.4 with unscaled dot products, then Eq.5 on already-expanded features. The expansion tensor below is PROVIDED test data, not a replacement for the full paper’s learned expansion.''')
    task(['prompt_weights','prompt_reduce'],'''# TODO — Trompt Eq.4 and Eq.5.
def prompt_weights(fused_prompts, columns):
    return ____

def prompt_reduce(weights, expanded_features):
    return ____''')
    code('''# CHECK — non-square axes and controlled row-specificity.
columns=torch.tensor([[1.,0.],[0.,1.],[0.,0.]])
shared=torch.zeros(2,2,2)  # B=2, P=2, d=2; same initial fused query
w=prompt_weights(shared,columns)
assert w.shape==(2,2,3) and torch.allclose(w.sum(-1),torch.ones(2,2))
assert torch.equal(w[0],w[1]), 'Identical fused queries must produce identical weights.'
changed=shared.clone();changed[1,0,0]=2
wc=prompt_weights(changed,columns)
assert torch.allclose(wc[1,0],torch.tensor([.786986,.106507,.106507]),atol=1e-6), 'Eq.4 has no sqrt(d) divisor.'
values=torch.tensor([2.,4.,9.]).reshape(1,1,3,1).expand(2,2,3,2)
z=prompt_reduce(wc,values)
assert z.shape==(2,2,2), 'Sum columns C; preserve prompts P and coordinates d.'
assert abs(z[0,0,0].item()-5)<1e-6 and abs(z[1,0,0].item()-2.958563)<1e-5
assert torch.equal(z[0],prompt_reduce(wc[:1],values[:1])[0]), 'No cross-row information route.'
print('CHECK 3 passed. Changed query reweights columns without any other batch row.')''')
    section('protocol');section('evidence');figure('scores','Author-reference test AUROC on three released tasks; seed points and ± sample SD. These are not current kernel outputs.')
    md('''### PROVIDED — inspect data provenance before training
The authors’ already-prepared arrays are deliberately used unchanged. Their upstream preprocessing fit scope is not independently reconstructed. We fit only the new MI order on training rows. Checksums detect changed downloaded data; they do not prove absence of upstream leakage.''')
    code('''data=paper_data('pima')
print({k:len(data[k]) for k in ('train','valid','test')})
print('Columns in model order:',[data['meta']['columns'][i] for i in data['meta']['order']])
display(pd.DataFrame(data['x'][:5]))
assert not (set(data['train']) & set(data['test']))
assert data['importance'].tolist()==sorted(data['importance'].tolist(),reverse=True)
print('Declared source:',data['meta']['source'])''')
    md('''### PROVIDED — the comparison harness is visible too
It trains the model defined above. CatBoost is a library baseline; it is not the architecture being taught. The same split is held across arms; these different model families receive explicit fixed budgets, not equal compute or a paper-level search.''')
    code("ARMS=('excel_no_da','excel_feat_mix','catboost')\n"+extract(HARNESS,{'experiment'}))
    code('''# CHECK — prove that training reaches your function, not a packaged replacement.
original_spa=spa_attention
calls={'count':0}
def counted_spa(*args,**kwargs):
    calls['count']+=1
    return original_spa(*args,**kwargs)
spa_attention=counted_spa
try:
    torch.manual_seed(0)
    smoke=ExcelFormer(data['x'].shape[1],d=16,heads=2,layers=1,dropout=0)
    smoke,_=train_excel(smoke,data['x'],data['y'],data['train'],data['valid'],epochs=1)
finally:
    spa_attention=original_spa
assert calls['count']>0,'Training bypassed your attention implementation.'
print('Live student attention calls:',calls['count'])''')
    md('''### Run the three-task experiment
Before execution, predict where augmentation may help and why the tiniest test set may saturate. Changing that prediction after seeing the scores is not evidence of foresight. Seeds measure initialization/training variability on fixed splits, not population uncertainty.''')
    code('''# PROVIDED — measured outputs from YOUR current code.
local={}
for name in ('pima','breast','banknote'):
    local[name]=experiment(paper_data(name),model_cls=ExcelFormer,train_fn=train_excel,predict_fn=predict_excel)
    print('Finished',name)
summary=summarize({k:v['auroc'] for k,v in local.items()},ARMS)
def summary_table(summary):
    return pd.DataFrame({ds:{m:f"{v['mean']:.4f} ± {v['sample_std']:.4f}" for m,v in models.items()}
                         for ds,models in summary['per_dataset'].items()}).T
display(summary_table(summary))
print('Mean ranks:',summary['mean_ranks'],'Friedman:',summary.get('friedman'),'CD:',summary.get('nemenyi_cd'))''')
    code('''# CHECK — local reproducibility against author evidence on this software environment.
author=json.loads(Path('_verify_l049_results.json').read_text())
same_versions=environment()['versions']==author['environment']['versions']
max_score_change=max(abs(summary['per_dataset'][ds][m]['mean']-author['summary']['per_dataset'][ds][m]['mean']) for ds in local for m in ARMS)
print('Same recorded versions:',same_versions,'max mean AUROC change:',max_score_change)
if same_versions:
    assert max_score_change<1e-6,'Inspect student code / protocol before interpreting changed rankings.'
# Different package versions may produce different scores; document them, do not overwrite the ledger.
print('CHECK 4: evidence computed; compare full protocol, not one rounded score.')''')
    section('temporal');figure('transfer','Author-reference MovieLens transfer results; random and chronological test populations differ. These are not two independent benchmark datasets.')
    code('''# PROVIDED — replay time transfer with the same live implementation.
transfer={}
for kind in ('random','temporal'):
    frame=temporal_data(kind)
    ranges=frame['meta']['timestamp_ranges']
    if kind=='temporal':
        assert ranges['train'][1]<ranges['valid'][0]<ranges['test'][0], 'Timestamp groups crossed partitions.'
    transfer[kind]=experiment(frame,model_cls=ExcelFormer,train_fn=train_excel,predict_fn=predict_excel)
    print(kind,ranges)
transfer_summary=summarize({k:v['auroc'] for k,v in transfer.items()},ARMS)
display(summary_table(transfer_summary))
assert transfer['random']['protocol']['row_ids']==transfer['temporal']['protocol']['row_ids']
print('Same sampled events. Different test populations: do not call the score difference a paired causal effect.')''')
    md('''### EXIT — write a claim audit
Fill the four fields below using your evidence. The automatic check can verify that you supplied an answer; it cannot grade your reasoning. Paste the printed ticket to your teacher for feedback. A loss or INCOMPARABLE verdict is acceptable; inventing a replicated claim is not.''')
    answers={'paper_claim':'ExcelFormer v5 Table 14 Pima-Indians-Diabetes reports Feat-Mix default AUROC 0.8356 across five runs; Trompt instead reports scoped comparability with trees.',
             'local_scope':'Three released numeric tasks, fixed small architecture and budgets, three model seeds; rank test has little power and banknote saturates.',
             'temporal_limit':'Same 12000 MovieLens events but different test populations and a frequency-encoded feature pipeline; this cannot refute an IID benchmark claim.',
             'reproduction_verdict':'INCOMPARABLE to the paper table: forward fidelity is verified, but augmentation sampler, tuning, seed aggregation and upstream preprocessing remain gaps.'}
    code('# EXIT TICKET — teacher example\naudit='+repr(answers) if solution else "# TODO — your written audit; retain the dictionary keys.\naudit = {\n    'paper_claim': '____',\n    'local_scope': '____',\n    'temporal_limit': '____',\n    'reproduction_verdict': '____',\n}")
    code('''# CHECK / EXIT — content is reviewed by the teacher, not automatically scored.
assert all(isinstance(v,str) and '____' not in v and len(v.split())>=8 for v in audit.values()), 'Supply your own supported explanations.'
ticket={'lesson':49,'audit':audit,'local_summary':summary,'transfer_summary':transfer_summary,
        'environment':environment(),'full_trompt_benchmark':'NOT_RUN'}
Path('data/cache/l049-exit.json').write_text(json.dumps(ticket,indent=2))
print(json.dumps(ticket,indent=2))''')
    section('scale')
    md('''### NEXT STEP — run closer to the paper
The loop below reuses the same model and training functions you just read. Set the gate to True on a suitable runtime. `smoke` verifies the operator quickly; `closer` uses width 256 / 3 blocks / 32 heads on Pima; `paper` increases caps but still does not run the published suite or tuning search. Keep those gaps in the verdict.

The output directory stores a contract, per-seed results, predictions and selected weights. Use a fresh directory after implementation or protocol changes. For unattended GPU execution: `modal run --detach modal/l049_paper_repro.py --preset closer` from the repository root. No cloud job has been launched by authoring this lesson.''')
    # Constants and function are visible; no import of a hidden scale-up model.
    import ast
    tree=ast.parse(REPRO);node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='PRESETS' for t in n.targets))
    code('# PROVIDED — explicit presets\n'+ast.get_source_segment(REPRO,node))
    code('# PROVIDED — resumable closer-to-paper runner\n'+extract(REPRO,{'run'}))
    code('''# NEXT STEP — gated OFF until you choose a runtime and output directory.
RUN_PAPER_REPRO = False
PAPER_PRESET = 'closer'
PAPER_OUT = 'data/cache/l049-student-closer'
if RUN_PAPER_REPRO:
    paper_result=run(PAPER_PRESET,PAPER_OUT,model_cls=ExcelFormer,train_fn=train_excel,
                     predict_fn=predict_excel,attention_fn=spa_attention,mix_fn=feat_mix)
    display(summary_table(paper_result['summary']))
    print(paper_result['verdict'],paper_result['gaps'])
else:
    print('Current student scale-up: NOT_RUN. Author larger run is a separate snapshot.')''')
    md('''**Ask a follow-up.** Explain any unclear arrow, tensor axis, source discrepancy, seed interval or claim boundary. Bring your EXIT ticket; completion alone does not establish mastery of every introduced term. Next lesson is the Q1 fair-comparison checkpoint.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    return nb

if __name__=='__main__':
    for solution in (False,True):
        path=HERE/('solutions' if solution else '')/'0049-excelformer-trompt.ipynb'
        path.parent.mkdir(exist_ok=True);nbf.write(build(solution),path);print(path)

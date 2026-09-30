"""Canonical HTML, reference, portable exercise notebooks and visible full implementation."""
import ast,base64,hashlib,io,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l144';S='0144-contextgnn';TITLE='ContextGNN: one graph, two ways to rank'
def definitions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
functions=definitions(P/'relkit/context_l144.py');checks=definitions(P/'_check_l144.py')
summary=json.loads((E/'summary.json').read_text()) if (E/'summary.json').exists() else {'pilots':[],'full_selected_reproduction':'INCOMPLETE','verified_ranking_rows':0}
# Keep the shallow comparator before ContextGNN in every rendered evidence packet.
summary['pilots'].sort(key=lambda row: (row['arm'] != 'shallowrhsgnn', row['arm']))
results='| Evidence | Validation MAP@10 | Test MAP@10 | Scope |\n|---|---:|---:|---|\n| Published ContextGNN | 28.55% | 28.02% | Paper Table 2 |\n| Published ShallowItem | 12.59% | 10.66% | Paper Table 2 |\n'
for r in summary['pilots']:
 results+=f"| {r['arm']} timing pilot | {100*r['scores']['val']:.3f}% (partial) | NOT_RUN | {r['history'][0]['queries']:,} training queries; first 1,024 validation rows |\n"
results+='\n**Full selected-task reproduction: '+summary['full_selected_reproduction']+'.** Pilot scores are not comparable with the paper or each other as a model-selection result. Historical identity: **NOT_ESTABLISHED**. Whole-paper reproduction: **NOT_RUN**.\n'
if (E/'cost-decision.json').exists():
 decision=json.loads((E/'cost-decision.json').read_text());results+=f"\nBudget decision: **{decision['decision']}**. {decision['explanation']}\n"
if (E/'independent-labels.json').exists():
 audit=json.loads((E/'independent-labels.json').read_text());results+=f"\nAn independent interval/join algorithm reconstructed **{sum(audit['independently_rebuilt_queries'].values()):,}** train/validation/test query labels. Saved pilot rankings independently rescored: **{summary['verified_ranking_rows']:,}**.\n"
results+='\nReal pilot first-batch gradients contain nonfinite entries: '+', '.join(r['arm']+': '+str(sum(r['first_batch_nonfinite_gradients'].values())) for r in summary['pilots'])+'. Finite output parity does not establish healthy optimization.\n'
results+='\nThe synthetic full-network fixture and original score-operator gradient comparison are mechanism evidence only. The stale RHS evaluation cache is independently demonstrated in that fixture.\n'
captions={'architecture':'Released ContextGNN: one temporal graph, contextual local scores and catalog-wide shallow item scores.','ownership':'Each local score replaces exactly one owner/item coordinate; another query’s score is untouched.','ranking':'Illustrative logits, not benchmark measurements. Sponsor C becomes less highly ranked after local replacement.','metric':'Compute precision at each relevant rank, divide by min(number of positives, k), then average across queries.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l144/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l144/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="context-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll the figure on a narrow screen.</figcaption></figure>')
 s=s.replace('[[WARMUP]]','Recall the three questions above before proceeding.' if portable else '<div id="warmup"></div>')
 s=s.replace('[[RANKING_WIDGET]]','Baseline scores [3, 2, 1, 4] rank D, A, B, C. Predict the order after adding +2 to local A and C, then verify with your fuse_scores function.' if portable else '<div id="context-ranking"></div><noscript>Baseline: scores [3,2,1,4]; ranking D → A → B → C. Add +2 only to local A/C: [5,2,3,4], ranking A → D → C → B.</noscript>')
 s=s.replace('[[TEACHBACK]]','Write your defense before checking the reference answer.' if portable else '<div id="context-teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="context-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','contextgnn-viz','contextgnn-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 144 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/contextgnn.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0143-relgnn-reproduction.html">Lesson 143</a></nav><header><p class="context-kicker">Year 4 · Quarter 3 · Lesson 144</p><h1>'+title+'</h1></header>'+html+'</article>'+''.join(f'<script src="../assets/{name}.js"></script>' for name in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## The score contract
Query=(facility, cutoff); candidate=sponsor. The released model forms tower scores from projected root and shallow sponsor embeddings, then replaces local (owner,item) entries with contextual head + root-item dot product + local offset. It does not add both branches. Root-specific membership matters.

## Temporal and metric contracts
Facility-study dates in (cutoff, cutoff+365days] define future sponsor targets via study IDs. Every sampled timestamp must be ≤ its owner cutoff. MAP@10 averages precision-at-hit sums divided by min(positive count,10). Align predictions by facility AND cutoff. Reject duplicate recommendations and missing query rows.

## Tensor trace
Root states B×h → projection B×d. Shallow sponsor vectors N×d → tower matrix B×N. Sampled contextual sponsor states M×h → M local scores. M owner/item index pairs choose replacement positions in B×N.

## Source versus paper
The released head includes root-item dot products and two offsets. Shallow item embeddings do not enter its GNN input. The release searches additional parameters and leaves the Optuna sampler unseeded. Evaluation RHS caches persist across epochs; source parity does not establish fresh evaluation embeddings.

## Reproduction evidence
'''+results+'''
## Recall without looking
Why can local replacement lower a score? Why is “outside this sampled graph” different from a new sponsor? Why can a source-matching run still have an evaluation-cache problem?

[Lesson](../lessons/0144-contextgnn.html) · [Protocol](../labs/l144-reproduction.md) · [Paper](https://arxiv.org/abs/2411.19513). Ask the teaching agent to review your implementation and evidence defense.
'''
(R/'reference/contextgnn.html').write_text(doc('ContextGNN reference',reference))
files={str(p.relative_to(P/'sources/l144')):p.read_text() for p in sorted((P/'sources/l144').rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
payload={}
for p in sorted(E.glob('pilot-*/*')):
 if p.name in ['predictions.npz','val-targets.json','result.json']:payload[str(p.relative_to(E))]=base64.b64encode(p.read_bytes()).decode()
packet='''# PROVIDED: compact pinned source and author evidence; not learner results.
SOURCE_FILES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())+''')))
SOURCE_ROOT=Path('sources/l144');SOURCE_ROOT.mkdir(parents=True,exist_ok=True)
for name,text in SOURCE_FILES.items():
    target=SOURCE_ROOT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
AUTHOR_FILES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode())+''')))
AUTHOR_SUMMARY='''+repr({k:summary[k] for k in ['pilots','verified_ranking_rows','full_selected_reproduction']})+'''
sys.path.insert(0,str(SOURCE_ROOT.resolve()))
'''
bootstrap='''# PROVIDED — default CPU fixture and saved-evidence audit; cloud training is separate.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cu124'])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==1.26.4','pandas==2.2.3','pyarrow==18.1.0','scipy==1.14.1','scikit-learn==1.5.2','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','sentence-transformers==3.3.1','transformers==4.46.3','huggingface-hub==0.26.5','duckdb==1.1.3','optuna==4.0.0'])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pyg_lib==0.4.0+pt25cu124','--find-links','https://data.pyg.org/whl/torch-2.5.1+cu124.html'])
    if 'torch' in sys.modules and sys.modules['torch'].__version__.split('+')[0]!='2.5.1':
        raise RuntimeError('Pinned packages installed. Restart the Colab session, then rerun from the top.')
from pathlib import Path
import ast,base64,hashlib,io,json,zlib,tempfile
import numpy as np
import torch
RUN_FULL_REPRODUCTION=False
'''
model=(P/'relkit/contextgnn_l144.py').read_text().replace('from relkit.context_l144 import fuse_scores','')
trainer=(P/'_full_l144.py').read_text();trainer=re.sub(r'^from relkit[^\n]*\n','',trainer,flags=re.M)
mechanism=definitions(P/'_mechanism_l144.py')['mechanism_check'].replace(' import relkit.contextgnn_l144 as visible','').replace('dict(vars(visible) if visible_namespace is None else visible_namespace)','dict(visible_namespace)')
scoring='''# CHECK: rescore shuffled author predictions using YOUR keyed_map.
rows=[]
for name,encoded in AUTHOR_FILES.items():
    if not name.endswith('/predictions.npz'):continue
    phase=name.split('/')[0]
    z=np.load(io.BytesIO(base64.b64decode(encoded)))
    keys=list(zip(z['val_entity'],z['val_time']))
    targets=json.loads(base64.b64decode(AUTHOR_FILES[phase+'/val-targets.json']))
    r=json.loads(base64.b64decode(AUTHOR_FILES[phase+'/result.json']))
    order=np.arange(len(keys))[::-1]
    score=keyed_map(keys,targets,[keys[i] for i in order],z['val_pred'][order],z['val_pred'].shape[1])
    assert abs(score-r['scores']['val'])<1e-12
    rows.append(dict(phase=phase,partial_validation_map=score,rows=len(keys)))
print(rows)
'''
gate='''# PROVIDED: full selected-task search. Read cost-decision.json before enabling.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for package,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1','optuna':'4.0.0'}.items():
        assert md.version(package).split('+')[0]==version,(package,'requires pinned GPU environment')
    assert torch.cuda.is_available(),'Requires pinned GPU environment and matching pyg-lib'
    root=Path('l144-prepared')
    prepare(root,SOURCE_ROOT)
    full_results=[full_search(root,Path('l144-full')/arm,arm) for arm in ['contextgnn','shallowrhsgnn']]
else:
    print('Full selected-task reproduction NOT_RUN by this notebook. Author timing pilots are separate.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 144 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student notebook')+' · PROVIDED / TODO / CHECK / EXIT. Three live functions: fusion, owner cutoffs, keyed MAP. Tier B full relational reproduction is separate from the Tier C CPU neural fixture. Default execution audits saved author evidence; it does not train the benchmark or establish mastery.'),nb.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  task=None
  if section.startswith('## 3'):task=('fuse_scores','check_fusion')
  if section.startswith('## 4'):task=('audit_cutoffs','check_cutoffs')
  if section.startswith('## 5'):task=('keyed_map','check_map')
  if task:
   name,check=task;code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
   cells.append(nb.v4.new_code_cell('# TODO — live implementation contract\n'+code));cells.append(nb.v4.new_code_cell('# CHECK — run unchanged\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · complete model, in execution order\n\nRead each class alongside the architecture. This is source-adapted MIT code with the local replacement delegated to your live function. Library primitives supply typed convolutions and column encoders. The full model, score construction, training loop, and selection behavior remain visible.'))
 for part in re.split(r'(?=^# SOURCE:)',model,flags=re.M):
  if part.strip():cells.append(nb.v4.new_code_cell(part))
 cells.append(nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']}))
 cells.append(nb.v4.new_markdown_cell('## CHECK · mechanism, gradients and stale-cache probe\n\nThe small synthetic relational fixture runs a complete neural forward/backward pass. The original score method is loaded from pinned source as a differential oracle. The CPU fixture has an explicit timestamp conversion for pandas 3 compatibility; the benchmark environment remains pinned to pandas 2.2.3.'))
 cells.append(nb.v4.new_code_cell(mechanism+'\nmechanism_report=mechanism_check(SOURCE_ROOT,globals())\nprint(mechanism_report)'))
 cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## NEXT STEP · full selected-task protocol\n\nThis GPU-only lane includes fresh feature preparation, full labels and 50 tuning trials plus five repeats per arm. It is off by default. Read the linked budget decision before running; a free Colab account does not establish runtime feasibility. Pinned packages and matched torch/pyg-lib are mandatory. See the protocol for exact Modal and isolated-environment commands.'))
 for part in re.split(r'(?=^def (?:sha|prepare|run_fit|full_search)\()',trainer,flags=re.M):
  if part.strip():cells.append(nb.v4.new_code_cell(part))
 cells.append(nb.v4.new_code_cell(gate))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',mechanism=mechanism_report,author_ranking_rows=sum(r['rows'] for r in rows),full_training='RUN' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l144-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 for i,c in enumerate(cells):c.id=f'l144-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});path=P/('solutions' if solution else '')/f'{S}.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  old=nb.read(path,4)
  if len(old.cells)==len(cells) and all(a.cell_type==b.cell_type and a.source==b.source for a,b in zip(old.cells,cells)):
   notebook=old
 nb.write(notebook,path)
print('Built L144 lesson, reference, student and solution')

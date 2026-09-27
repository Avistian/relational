"""Canonical HTML and portable notebooks with visible source and live tasks."""
import ast,base64,gzip,hashlib,io,json,re,textwrap
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l138';S='0138-ecommerce-amazon';TITLE='E-commerce: what does Amazon churn actually mean?'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
funcs=defs(P/'relkit/amazon_l138.py');checks=defs(P/'_check_l138.py');audit=json.loads((E/'task_audit.json').read_text());repro=json.loads((E/'reproduction.json').read_text())
table='| Split | Queries | Churn prevalence | Recency AUROC | Recency AP |\n|---|---:|---:|---:|---:|\n'
for split,r in audit['splits'].items():table+=f"| {split} | {r['rows']:,} | {r['positive_rate']:.4f} | {r['recency_auc']:.4f} | {r['recency_ap']:.4f} |\n"
table+='\n**Measured:** all **5,470,060** labels reconstructed with zero mismatches and zero ineligible queries. Both database and task archive hashes match the pinned release. The full raw archive contains 20,862,040 reviews, including later events needed for label reconstruction. The model snapshot excludes reviews after its test cutoff.'
status='**Neural reproduction status: '+repro['status']+'.** '+repro['reason']+'\n\nFull five-run training: **'+repro['full_training']+'**. Historical run identity: **NOT_ESTABLISHED**. See the [execution ledger](../labs/evidence/l138/reproduction.json) and [budget](../labs/_budget_l138.json).'
training=json.loads((E/'training.json').read_text()) if (E/'training.json').exists() else None
if training:
 status+='\n\n| Split | Fresh AUROC % ± sample SD | Paper AUROC % ± reported SD | Descriptive verdict |\n|---|---:|---:|---|\n'
 for split,paper in [('val','70.45 ± 0.06'),('test','70.42 ± 0.05')]:
  x=training['metrics'][split];status+=f"| {split} | {100*x['mean']:.4f} ± {100*x['sample_sd']:.4f} | {paper} | {x['verdict']} |\n"
 status+='\nAll five seeds completed ten source-defined epochs. All 3,808,385 held-out predictions were aligned to audited query identities and independently rescored. The predeclared 1.0 percentage-point tolerance is descriptive, not an equivalence test. Historical training identity and whole-paper reproduction remain unestablished.'
captions={'windows':'Source-exact endpoint contract: recent (−91,0], future (0,91]. Open circles exclude endpoints; filled circles include them.','architecture':'Basic RDL on customer, review and product rows. The highlighted two-hop path carries historical book information into the customer root.','ranking':'Synthetic scores: three wins and one tie yield AUROC 0.875. This is arithmetic, not a measured model score.'}
def prose(portable=False,student=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[AUDIT_RESULTS]]',table).replace('[[REPRO_RESULTS]]',status)
 for marker,name in [('TARGET_CODE','review_target'),('PATH_CODE','visible_paths'),('AUC_CODE','rank_auc')]:s=s.replace('[['+marker+']]','Implement this contract in the live TODO below.' if student else '```python\n'+funcs[name]+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l138/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l138/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for marker,id,fallback in [('WINDOW_WIDGET','l138-window','Baseline: day+91 gives churn 0; day+92 gives churn 1. Eligibility stays true.'),('PATH_WIDGET','l138-path','At cutoff0, only r0 and book0 enter. At cutoff1, r1 and book1 also enter.')]:s=s.replace('[['+marker+']]',fallback if portable else f'<div class="tuning-control" id="{id}"></div><noscript>{fallback}</noscript>')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write your defense, then ask the teacher to evaluate it.' if portable else '<div id="l138-teachback"></div>')
 if portable:s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,s,interactive=False):
 body=render(s).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','amazon-task-viz'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 138 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/tuning-budget.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0137-error-analysis-reg.html">Lesson 137</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Lesson 138</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Query contract

A query is (customer, cutoff). Eligible iff at least one review occurs in (t − 91 days,t]. Churn 1 iff none occurs in (t,t + 91 days]. Ineligible means no query, not churn 0 or churn 1. Finalize labels only after the complete horizon.

## Information contract

Rows become nodes; foreign keys become bidirectional graph links. Two layers let a product influence a customer through a review. Apply the same query cutoff at every hop. Timestamp-less metadata has a separate availability limitation. Released feature statistics use the supplied snapshot, not train-only queries.

## Score contract

Higher scores mean more predicted review absence. AUROC counts positive-negative wins plus half of ties, divided by the number of pairs. Both classes must exist. AP also depends on prevalence. Neither metric identifies purchasing churn or intervention effects.

## Evidence contract

The named neural target is RelBench v1 Table 6 user-churn: five runs, validation 70.45±.06 / test 70.42±.05 AUROC percentage points. Full label reconstruction, a recency baseline, neural fixtures, preprocessing timings and completed model fits establish different things. Preserve that distinction. The released training table has 24,172 fewer rows than the paper; complete execution therefore establishes released-pipeline replay, while exact historical reproduction remains unestablished.

[Lesson](../lessons/0138-ecommerce-amazon.html) · [Protocol](../labs/l138-reproduction.md) · [Source SQL](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/tasks/amazon.py)
'''
(R/'reference/ecommerce-amazon.html').write_text(doc('Amazon task field guide',reference))
bootstrap='''# PROVIDED: portable defaults; full training is OFF.
import importlib.util,subprocess,sys
for module,package in [('numpy','numpy'),('pandas','pandas'),('torch','torch'),('torch_frame','pytorch-frame==0.3.0'),('torch_geometric','torch-geometric'),('relbench','relbench==1.1.0')]:
    if importlib.util.find_spec(module) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',package])
import numpy as np, pandas as pd, json, hashlib, base64, gzip, io
from pathlib import Path
RUN_FULL_REPRODUCTION=False
'''
raw=(E/'samples.json').read_bytes();sample_code='''# PROVIDED: checksum-pinned real queries; only IDs and relative event days.
raw=gzip.decompress(base64.b64decode('''+repr(base64.b64encode(gzip.compress(raw,mtime=0)).decode())+'''))
assert hashlib.sha256(raw).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
SAMPLES=json.loads(raw)
for split,rows in SAMPLES.items():
    for row in rows:
        assert review_target(row['events'],0)==(True,row['target'])
print('PASS: 36 real queries independently reconstructed with YOUR function')
'''
# Narrow portable packet: all held-out scores and labels, not a huge Python literal.
z=np.load(E/'recency_predictions.npz');buf=io.BytesIO();np.savez_compressed(buf,**{k:z[k] for k in z.files if k.endswith(('_target','_score'))});raw=buf.getvalue()
score_code='''# PROVIDED: all held-out recency predictions; no retraining implied.
raw=base64.b64decode('''+repr(base64.b64encode(raw).decode())+''')
assert hashlib.sha256(raw).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
SCORES=np.load(io.BytesIO(raw));EXPECTED='''+repr({k:v['recency_auc'] for k,v in audit['splits'].items() if k!='train'})+'''
score_rows=[]
for split in ['val','test']:
    value=rank_auc(SCORES[split+'_target'],SCORES[split+'_score'])
    assert abs(value-EXPECTED[split])<1e-12
    score_rows.append(dict(split=split,queries=len(SCORES[split+'_target']),auc=value))
display(pd.DataFrame(score_rows))
'''
model=(P/'relkit/rdl_l117.py').read_text().split('# %% Full released model dependencies (PROVIDED)',1)[1].split('# %% Full released-protocol training loop (PROVIDED)',1)[0]
model='\n'.join('# '+line for line in (P/'sources/l138/LICENSE').read_text().splitlines())+'\n'+model
full=(P/'_full_l138.py').read_text().split("if __name__=='__main__':",1)[0].replace('from relkit.rdl_l117 import Model,make_pkey_fkey_graph,get_node_train_table_input\n','')
fixture=(P/'_fixture_l138.py').read_text().split("if __name__=='__main__':",1)[0]
gate='''# NEXT STEP: full source protocol. Check cost/memory before enabling.
# Local/Colab has no billing guard; Modal's operator enforces the aggregate cap.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    required={'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}
    for name,version in required.items():assert md.version(name).split('+')[0]==version,(name,'use requirements-l117-runtime.txt and matching torch/pyg-lib')
    assert torch.cuda.is_available(),'A GPU and sufficient host memory are required'
    for seed in range(5):full_run(Path('l138-full')/f'seed-{seed}',seed=seed,epochs=10,cache='l138-materialized')
else:print('Full five-seed reproduction NOT_RUN in this default notebook')
'''
neural_payload=None
if training:
 arrays={}
 for seed in range(5):
  a=np.load(E/'final'/f'seed-{seed}'/'predictions.npz')
  for split in ['val','test']:arrays[f'{split}_{seed}']=a[split+'_pred']
 buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue()
 neural_payload='# PROVIDED: compressed complete neural prediction vectors. Expand only to inspect the data transport.\nNEURAL_RAW=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\nassert hashlib.sha256(NEURAL_RAW).hexdigest()=='+repr(hashlib.sha256(raw).hexdigest())
 neural_check="# CHECK: source audit aligned entity/cutoff keys before packaging.\n# This cell re-scores the preserved ordering using YOUR rank_auc implementation.\nNEURAL=np.load(io.BytesIO(NEURAL_RAW));neural_rows=[]\nfor split in ['val','test']:\n    values=[]\n    for seed in range(5):\n        pred=NEURAL[f'{split}_{seed}'];y=SCORES[split+'_target']\n        assert len(pred)==len(y)\n        value=rank_auc(y,pred);values.append(value)\n        neural_rows.append(dict(split=split,seed=seed,auc=value))\n    assert abs(float(np.mean(values))-NEURAL_EXPECTED[split])<1e-12\ndisplay(pd.DataFrame(neural_rows))\nprint('PASS: all 3,808,385 archived neural predictions re-scored; no fresh training implied')\n"
 neural_check='NEURAL_EXPECTED='+repr({k:v['mean'] for k,v in training['metrics'].items()})+'\n'+neural_check

for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 138 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED: inspect. TODO: implement. CHECK: run unchanged. EXIT: defend. Default execution checks real labels, re-scores 761,677 recency predictions' + (' and 3,808,385 neural predictions' if training else '') + ' and runs a synthetic neural fixture. Fresh full training is separately gated.'),nb.v4.new_code_cell(bootstrap)]
 sections=re.split(r'(?=^## )',prose(True,not solution),flags=re.M)
 for section in sections:
  cells.append(nb.v4.new_markdown_cell(section))
  if section.startswith('## 7') and neural_payload:
   cells.append(nb.v4.new_markdown_cell('### CHECK · all measured neural predictions\n\nThe author audit checked entity/cutoff alignment for every seed. This compressed packet preserves that ordering; your live metric now reconstructs all ten held-out scores. The data-transport cell is collapsed; the scoring code stays visible.'))
   cells.append(nb.v4.new_code_cell(neural_payload,metadata={'tags':['data-payload'],'jupyter':{'source_hidden':True}}))
   cells.append(nb.v4.new_code_cell(neural_check))
  task=None
  if section.startswith('## 2'):task=('review_target','check_target')
  if section.startswith('## 4'):task=('visible_paths','check_paths')
  if section.startswith('## 6'):task=('rank_auc','check_auc')
  if task:
   name,check=task;signature=funcs[name].splitlines()[0]
   cells.append(nb.v4.new_code_cell('# '+('SOLUTION' if solution else 'TODO')+'\n'+(funcs[name] if solution else signature+'\n    raise NotImplementedError("TODO: '+name+'")')))
   cells.append(nb.v4.new_code_cell('# CHECK\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+' contract")'))
   if name=='review_target':cells.append(nb.v4.new_code_cell(sample_code))
   if name=='rank_auc':cells.append(nb.v4.new_code_cell(score_code))
 cells += [nb.v4.new_markdown_cell('## PROVIDED · complete visible released model and graph construction\n\nThis code uses PyTorch Frame column encoders and PyG graph primitives. The synthetic fixture below executes the complete forward/backward path, separately from the full-data experiment.'),nb.v4.new_code_cell(model),nb.v4.new_code_cell(fixture+'\nFIXTURE,_,_,_=neural_fixture(Model)\nprint(FIXTURE)'),nb.v4.new_markdown_cell('## NEXT STEP · complete source trainer\n\nThe full runner includes text embeddings, archive checks, sampling, optimization, validation selection and keyed prediction artifacts. It is not executed by the default notebook. Large host memory is needed; respect the reproduction budget. [Pinned environment](https://avistian.github.io/relational/labs/requirements-l117-runtime.txt).'),nb.v4.new_code_cell(full),nb.v4.new_code_cell(gate),nb.v4.new_code_cell("report=dict(status='PASS',real_query_examples=36,heldout_predictions=761677,neural_predictions=3808385 if 'NEURAL' in globals() else 0,neural_fixture=FIXTURE['status'],full_training='EXECUTED' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l138-report.json').write_text(json.dumps(report,indent=2));print(report)")]
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 for i,c in enumerate(notebook.cells):c.id=f'l138-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb');path.parent.mkdir(exist_ok=True)
 if path.exists() and solution:
  old=nb.read(path,4);a=[c for c in notebook.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   notebook.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nb.write(notebook,path)
print('Built lesson, reference and portable student/solution')

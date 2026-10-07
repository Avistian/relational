"""Canonical lesson and portable notebooks: source, live contracts and evidence."""
import ast,base64,hashlib,io,json,re
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l139';S='0139-healthcare-trial';TITLE='Clinical trials: transfer the pipeline, audit the target'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
funcs=defs(P/'relkit/trial_l139.py');checks=defs(P/'_check_l139.py')
audit=json.loads((E/'task_audit.json').read_text());training=json.loads((E/'training.json').read_text());repro=json.loads((E/'reproduction.json').read_text())
table='| Split | Audited queries | Positive labels | Prevalence |\n|---|---:|---:|---:|\n'
for k,v in audit['splits'].items():table+=f"| {k} | {v['rows']:,} | {v['positives']:,} | {v['prevalence']:.4f} |\n"
table+='\nFull audit: **'+str(sum(x['rows'] for x in audit['splits'].values()))+' queries**, zero label/key mismatches after applying the released split contract. '+audit.get('boundary_note','')
status='**'+repro['status'].replace('_',' ').capitalize()+'** — '+repro['reason']+'\n\n| Split | Fresh AUROC % ± sample SD | Paper AUROC % ± SD | Descriptive comparison |\n|---|---:|---:|---|\n'
for k,paper in [('val','68.18 ± .49'),('test','68.60 ± 1.01')]:
 v=training['metrics'][k];status+=f"| {k} | {100*v['mean']:.4f} ± {100*v['sample_sd']:.4f} | {paper} | {v['verdict']} |\n"
status+=f"\nAll {training['verified_predictions']:,} held-out neural predictions independently aligned and re-scored. {training['audited_query_occurrences']:,} sampled query occurrences audited. Five full20-epoch fits; no post-test tuning. The ±1pp closeness rule is descriptive, not equivalence. Historical identity and whole-paper reproduction remain **NOT_ESTABLISHED**."
captions={'selection':'Measured validation AUROC for all five complete fits. Dots mark first maxima; test scores play no role in checkpoint selection.','eligibility':'The clinical target has two decisions: whether a query exists, then whether any qualifying primary analysis has p≤.05.','architecture':'Facility → association → study, within the full clinical graph. Mean neighbors, sum relations, owner-specific time, study head and training objective.','transfer':'Shared model computation transfers; query semantics, schema and paper settings are audited afresh.'}
def prose(portable=False,student=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[AUDIT_RESULTS]]',table).replace('[[REPRO_RESULTS]]',status)
 for marker,name in [('TARGET_CODE','trial_target'),('VISIBILITY_CODE','visibility_mask'),('SCORE_CODE','keyed_auc')]:s=s.replace('[['+marker+']]','Implement this function in the live TODO below.' if student else '```python\n'+funcs[name]+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l139/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l139/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} On a narrow screen, scroll the diagram horizontally.</figcaption></figure>')
 for marker,id,fallback in [('TARGET_WIDGET','l139-target','Baseline day365: eligible, label1. Day366: absent query, not label0.'),('CUTOFF_WIDGET','l139-cutoff','Baseline association time15: hidden for cutoff10, visible for cutoff20.')]:s=s.replace('[['+marker+']]',fallback if portable else f'<div class="tuning-control" id="{id}"></div><noscript>{fallback}</noscript>')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write your four-paragraph defense and ask the teacher for feedback.' if portable else '<div id="l139-teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','trial-task-viz'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 139 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/tuning-budget.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0138-ecommerce-amazon.html">Lesson 138</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Lesson 139</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Population and target
A study must start by cutoff t and have at least one qualifying primary analysis in (t,t+365days]. Keep numeric p in[0,1], exclude modifier >; preserve other modifiers as the numeric source does. No qualifying result → no query. Minimum p≤.05 →1; otherwise0. This observes recorded statistical analyses, not causal clinical benefit.

## Query and graph
A key is(study,cutoff). A facility reaches a study through its association row in two hops. Every sampled timestamp uses its query owner; a batch maximum leaks. Outcome dates are inferred from study completion; several association dates from study start. Neither establishes publication/arrival time. The retrospective actual-completion cohort, untimestamped attributes and snapshot-fitted preprocessing remain separate limitations.

## Released trial settings
Two128-channel layers, mean neighbors, sum relations, batch512, uniform64/32 fanout; Adam.0001, BCE,20epochs, first strict validation AUROC maximum. Five fresh seeds0–4. Model code structure transfers, weights do not.

## Evidence and next use
'''+table+'\n\n'+status+'''\n\nPreprocessing reused from verified L132 artifacts; full fresh materialization path supplied. No whole-paper or historical RNG claim. Source/model checks do not establish learner mastery, live Colab or deployment. [Lesson](../lessons/0139-healthcare-trial.html) · [Protocol](../labs/l139-reproduction.md) · [Paper](https://arxiv.org/html/2407.20060v1).
'''
(R/'reference/healthcare-trial.html').write_text(doc('Clinical trial transfer — reference',reference))
bootstrap='''from pathlib import Path
import base64,hashlib,io,json
import numpy as np
import pandas as pd
from IPython.display import display
RUN_FULL_REPRODUCTION=False
PREPARED_ROOT=None  # Optional verified materialization directory; default prepares afresh.
# Enable only in the pinned GPU runtime, with sufficient host memory and budget.
'''
raw=(E/'samples.json').read_bytes();sample_code='RAW=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\nassert hashlib.sha256(RAW).hexdigest()=='+repr(hashlib.sha256(raw).hexdigest())+'''
SAMPLES=json.loads(RAW)
for rows in SAMPLES.values():
    for row in rows:assert trial_target(row['start'],row['analyses'],0)==(True,row['target'])
print('PASS: real trial queries reconstructed with your target function')
'''
arrays={};queries=np.load(E/'queries.npz')
for key in queries.files:arrays[key]=queries[key]
for seed in range(5):
 z=np.load(E/'full'/f'seed-{seed}'/'predictions.npz')
 for key in z.files:arrays[f's{seed}_{key}']=z[key]
trace=np.load(E/'full/seed-0/sampled_trace.npz')
for key in trace.files:arrays['trace_'+key]=trace[key]
buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue();packet='PACKET=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\nassert hashlib.sha256(PACKET).hexdigest()=='+repr(hashlib.sha256(raw).hexdigest())+'\nDATA=np.load(io.BytesIO(PACKET))\n'
score_code='EXPECTED='+repr({k:v['mean'] for k,v in training['metrics'].items()})+'''
# Your visibility function audits a real sampled training batch.
node_occurrences=0
for key in DATA.files:
    if key.startswith('trace_') and key.endswith('_times'):
        owners=DATA[key[:-6]+'_owners'];times=DATA[key]
        assert visibility_mask(times,owners,DATA['trace_cutoffs']).all()
        node_occurrences+=len(times)
print('PASS: real sampled timestamped nodes',node_occurrences)
rows=[];rng=np.random.default_rng(139)
for split in ['val','test']:
    keys=list(zip(DATA[split+'_study'],DATA[split+'_time']));targets=DATA[split+'_target'];values=[]
    for seed in range(5):
        prefix=f's{seed}_{split}_'
        assert np.array_equal(targets,DATA[prefix+'target'])
        pkeys=list(zip(DATA[prefix+'study'],DATA[prefix+'time']))
        order=rng.permutation(len(pkeys))
        auc=keyed_auc(keys,targets,[pkeys[i] for i in order],DATA[prefix+'pred'][order])
        values.append(auc);rows.append(dict(split=split,seed=seed,queries=len(keys),auc=auc))
    assert abs(np.mean(values)-EXPECTED[split])<1e-12
display(pd.DataFrame(rows))
print('PASS: all ten full held-out score vectors independently keyed and rescored')
'''
model=(P/'relkit/rdl_l117.py').read_text().split('# %% Full released model dependencies (PROVIDED)',1)[1].split('# %% Full released-protocol training loop (PROVIDED)',1)[0]
model='\n'.join('# '+line for line in (P/'sources/l139/LICENSE').read_text().splitlines())+'\n'+model
fixture=(P/'_fixture_l139.py').read_text().split("if __name__=='__main__':",1)[0]
full=(P/'_full_l139.py').read_text().split("if __name__=='__main__':",1)[0].replace('from relkit.rdl_l117 import Model,make_pkey_fkey_graph,get_node_train_table_input\n','')
gate='''# Full protocol: fresh materialization and five new fits; no hidden row caps.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    required={'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}
    for name,version in required.items():assert md.version(name).split('+')[0]==version,(name,'use pinned runtime')
    assert torch.cuda.is_available(),'Use a GPU with sufficient host memory'
    prepared=materialize(Path('l139-prepared')) if PREPARED_ROOT is None else Path(PREPARED_ROOT)
    for seed in range(5):full_run(Path('l139-full')/f'seed-{seed}',seed=seed,epochs=20,prepared_root=prepared)
else:print('Fresh full training NOT_RUN in default notebook; archived scores checked above')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 139 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED: inspect; TODO: implement; CHECK: run unchanged; EXIT: defend. Default execution audits a portable evidence packet and a synthetic full neural path. Fresh training is separately gated.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell('# PROVIDED: tie-aware rank arithmetic\n'+funcs['rank_auc'])]
 for section in re.split(r'(?=^## )',prose(True,not solution),flags=re.M):
  cells.append(nb.v4.new_markdown_cell(section));task=None
  if section.startswith('## 2'):task=('trial_target','check_target')
  if section.startswith('## 4'):task=('visibility_mask','check_visibility')
  if section.startswith('## 6'):task=('keyed_auc','check_score')
  if task:
   name,check=task;cells.append(nb.v4.new_code_cell('# '+('SOLUTION' if solution else 'TODO')+'\n'+(funcs[name] if solution else funcs[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")')))
   cells.append(nb.v4.new_code_cell('# CHECK\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")'))
   if name=='trial_target':cells.append(nb.v4.new_code_cell(sample_code,metadata={'tags':['data-payload']}))
  if section.startswith('## 7'):
   cells.append(nb.v4.new_code_cell(packet,metadata={'tags':['data-payload'],'jupyter':{'source_hidden':True}}));cells.append(nb.v4.new_code_cell(score_code))
 cells += [nb.v4.new_markdown_cell('## PROVIDED · complete model and graph construction\n\nRead the encoder, owner-specific temporal encoding, relation convolution, root selection and head. The fixture executes the whole neural path; it is separate from the full-data score experiment.'),nb.v4.new_code_cell(model),nb.v4.new_code_cell(fixture+'\nFIXTURE,_,_,_=neural_fixture(Model)\nprint(FIXTURE)'),nb.v4.new_markdown_cell('## NEXT STEP · full source training\n\n[Runtime](https://avistian.github.io/relational/labs/requirements-l117-runtime.txt). The default environment must provide PyTorch Frame, PyG, RelBench and their dependencies. The full lane additionally needs a GPU, matching pyg-lib and substantial host memory. Modal enforces the author budget; enabling this notebook gate does not enforce billing.'),nb.v4.new_code_cell(full),nb.v4.new_code_cell(gate),nb.v4.new_code_cell("report=dict(status='PASS',real_examples=sum(map(len,SAMPLES.values())),rescored_predictions=sum(r['queries'] for r in rows),neural_fixture=FIXTURE['status'],fresh_training='EXECUTED' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l139-report.json').write_text(json.dumps(report,indent=2));print(report)")]
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l139-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);a=[c for c in cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   notebook.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nb.write(notebook,path)
print('Built L139 lesson, reference, portable student and solution')

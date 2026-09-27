"""Build aligned HTML and portable notebooks from canonical source and evidence."""
import ast,base64,hashlib,json,re,textwrap
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0128-task-taxonomy';TITLE='Task taxonomy: choose the target, head, loss and metric'
canonical=(P/'relkit/taxonomy_l128.py').read_text()
functions={n.name:ast.get_source_segment(canonical,n) for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef)}
check_source=(P/'_check_l128.py').read_text();checks={n.name:ast.get_source_segment(check_source,n) for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef)}
tasks={'task_contract':'check_contract','binary_auc':'check_auc','ranking_map':'check_map'}
summary=json.loads((P/'evidence/l128/summary.json').read_text())
results='**Author execution: COMPLETE selected experiment with reconstructed historical labels.**\n\n| Split | Fresh mean AUROC (%) | Sample SD (pp) | Paper mean (%) | Verdict |\n|---|---:|---:|---:|---|\n'
for split in ['val','test']:
 m=summary['metrics'][split];results+=f"| {split} | {100*m['mean']:.4f} | {100*m['sample_sd']:.4f} | {100*m['target']:.2f} | {m['verdict']} |\n"
results+='\n| Seed | Selected epoch | Final validation AUROC (%) | Test AUROC (%) |\n|---|---:|---:|---:|\n'
for row in summary['seeds']:results+=f"| {row['seed']} | {row['selected_epoch']} | {100*row['val']:.4f} | {100*row['test']:.4f} |\n"
results+=f"\nAll **{summary['evaluated_queries']:,}** final predictions independently rescored; **{summary['temporal_audit']['audited_query_occurrences']:,}** sampled query occurrences audited. Maximum original-model raw-output difference **{summary['maximum_original_output_error']:.3g}**. Successful pilot plus five fits: **USD {summary['worker_resource_usd']:.6f}** worker estimate, excluding setup/storage/unitemized overhead. Failed first pilot conservatively reserves another **USD {summary['failed_pilot_resource_bound_usd']:.6f}**. Seven of eight one-hour worker reservations consumed. This is not an invoice total. [Full evidence](../labs/evidence/l128/summary.json).\n"
captions={'heads':'Illustrative values distinguish a probability, a numerical prediction and ranked candidate IDs. The head shape alone does not identify the task.', 'architecture':'Actual F1 binary-classification path: typed row encoders, relative time, two sum-GraphSAGE layers, seed-only logits, BCE training and validation AUROC selection.', 'metrics':'Synthetic worked arithmetic. AUROC averages four pair credits; AP averages precision contributions at relevant ranks, with the stated denominator.', 'time':'Future binary prediction produces one answer; an autoregressive decoder feeds its first generated output into the next prediction. No autoregressive model is run.', 'scores':'Five fresh complete runs with reconstructed historical labels. Dots are seeds; diamonds show mean and sample seed SD. Dashed lines are paper means; axes use detail scales.'}
fallback='Static metric trace: threshold changes the example accuracy but AUROC stays .875. AP@3 for [2,7,4] is .833333; [7,2,4] gives .583333; [2,4,7] gives 1.'
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 train=(P/'relkit/classification_l128.py').read_text();start=train.index('    for epoch in range(1,epochs+1):');end=train.index('    model.load_state_dict(state)',start)
 # Student markdown explains the mechanism without leaking the three TODO bodies.
 text=text.replace('[[TRAIN_CODE]]','```python\n'+textwrap.dedent(train[start:end])+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l128/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l128/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 text=text.replace('[[METRIC_WIDGET]]',fallback if portable else '<div id="l128-auc"></div><div id="l128-map"></div><noscript><p>'+fallback+'</p></noscript>')
 text=text.replace('[[WARMUP]]','Recall task-row identity and seed-only supervision before reading.' if portable else '<div id="warmup"></div>')
 text=text.replace('[[TEACHBACK]]','Write your explanation before comparing the evidence.' if portable else '<div id="l128-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','task-table','rdl-stack-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0127-relbench-v1.html">Lesson 127</a></nav><header><p class="stream-kicker">Year 4 · Quarter 1 · Lesson 128</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','task-metrics-viz','l128-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0127-relbench-v1.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source).replace("'pooch':'pooch'","'pooch':'pooch','sklearn':'scikit-learn'")
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l127','l128')
gate=gate.replace("task=get_task('rel-f1','driver-position',download=True)","task,recovery=historical_task(Path('l128-full'))")
gate=gate.replace(",('tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e')",'')
gate=gate.replace('frozen=experiment_contract()',"frozen=dict(epochs=10,seeds=list(range(5)))")
payload={}
for seed in range(5):
 root=P/f'evidence/l128/paper/seed-{seed}';raw=(root/'predictions.npz').read_bytes();payload[str(seed)]=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),result=json.loads((root/'result.json').read_text()))
real_harness="""# PROVIDED: rescore ALL real author probabilities using YOUR binary_auc.
import io,base64,hashlib,json,statistics
from pathlib import Path
import numpy as np
payload = """+repr(payload)+"""
expected = """+repr(summary['metrics'])+"""
records=[];count=0
assert task_contract('binary')['maximize']
for seed in range(5):
 item=payload[str(seed)];r=item['result'];raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 arrays=np.load(io.BytesIO(raw),allow_pickle=False)
 selected=max(r['trace'],key=lambda row:row['val_auc'])['epoch'];assert selected==r['selected_epoch']
 scores={}
 for split,n in [('val',566),('test',702)]:
  p=arrays[split+'_pred'];y=arrays[split+'_target'];keys=list(zip(arrays[split+'_entity'],arrays[split+'_time']))
  assert len(p)==len(y)==len(set(keys))==n and np.isfinite(p).all()
  score=binary_auc(y,p);assert abs(score-r['scores'][split])<1e-12
  assert abs(binary_auc(1-y,1-p.astype(np.float64))-score)<1e-12
  scores[split]=score;count+=n
 records.append(scores)
 print('Seed',seed,'selected epoch',selected,'AUROC',scores)
aggregate={s:dict(mean=statistics.mean(r[s] for r in records),sample_sd=statistics.stdev(r[s] for r in records)) for s in ['val','test']}
for s in aggregate:
 for k in aggregate[s]:assert abs(aggregate[s][k]-expected[s][k])<1e-12
report=dict(status='PASS',predictions=count,aggregate=aggregate,scope='Author evidence replay, not new training; historical labels reconstructed')
Path('l128-report.json').write_text(json.dumps(report,indent=2));print(report)
"""
# Portable complete archived task table, not a handful of synthetic rows.
raw=(P/'sources/l128/driver-dnf.zip').read_bytes()
label_harness="""# PROVIDED: inspect every released task row and historical label polarity.
import zipfile
import pandas as pd
raw=base64.b64decode("""+repr(base64.b64encode(raw).decode())+""")
assert hashlib.sha256(raw).hexdigest()=='bd562529a3c0016363d5cae247979fd3712d77ba948574f135dab88c036d3e2d'
label_counts={}
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 for split,n,pos in [('train',11411,1365),('val',566,125),('test',702,207)]:
  df=pd.read_parquet(io.BytesIO(z.read('driver-dnf/'+split+'.parquet')))
  assert len(df)==n and not df.duplicated(['driverId','date']).any()
  historical=1-df.did_not_finish
  assert historical.sum()==pos
  label_counts[split]=dict(queries=n,current_positive=int(df.did_not_finish.sum()),historical_positive=int(historical.sum()))
  if split!='train':
   for seed in range(5):
    a=np.load(io.BytesIO(base64.b64decode(payload[str(seed)]['base64'])))
    np.testing.assert_array_equal(a[split+'_target'],historical.to_numpy())
    np.testing.assert_array_equal(a[split+'_entity'],df.driverId.to_numpy())
    np.testing.assert_array_equal(a[split+'_time'],df.date.astype('int64').to_numpy())
print(pd.DataFrame(label_counts).T)
print('This cell verifies archive/prediction identity. Separate original-SQL and raw-event audits establish the historical transformation.')
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 128 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · PROVIDED = inspect; TODO = implement; CHECK = run; EXIT = defend. Default execution replays complete author predictions and task tables. It does not train a model. Full training is explicitly gated below. PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.extend([nbf.v4.new_markdown_cell('## Implement the task contract and metrics\n\nHints: time is a separate axis; tied positive-negative comparisons earn half credit; MAP normalizes inside each query before averaging.'),nbf.v4.new_code_cell('import math,json\n'+checks['rejected'])])
 descriptions={'task_contract':'Return fresh contract metadata. The full classification trainer calls this function; verify loss and selection direction.', 'binary_auc':'Compute all positive-negative ordering credits, validate inputs, and average. The real evidence replay below calls your function.', 'ranking_map':'Keep query boundaries and ranked order. Validate unique IDs; use the pinned normalization and empty-query convention.'}
 for name,check in tasks.items():
  src=functions[name] if solution else functions[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
  cells.extend([nbf.v4.new_markdown_cell('### TODO · '+name+'\n\n'+descriptions[name]),nbf.v4.new_code_cell(src),nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")')])
 cells.extend([nbf.v4.new_markdown_cell('## REPLAY · every author validation and test prediction\n\nUse your AUROC implementation and check inversion symmetry. These are reconstructed historical targets, not literal current DNF probabilities.'),nbf.v4.new_code_cell(real_harness),nbf.v4.new_markdown_cell('## TRACE · every task-table row\n\nThe complete downloaded archive is embedded, hash-checked and compared with all saved predictions.'),nbf.v4.new_code_cell(label_harness)])
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · complete model and trainer\n\nMIT-licensed source pinned to commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. Shared architecture is exact to the included upstream definitions. Historical label SQL is separately pinned before the label-flip commit. [Provenance](https://avistian.github.io/relational/labs/sources/l128/manifest.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1)
  if heading.startswith('Full released-protocol training'):continue
  cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 trainer=(P/'relkit/classification_l128.py').read_text();trainer='\n'.join(line for line in trainer.splitlines() if not line.startswith('from relkit.'))
 cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · classification trainer\n\nCalls your contract and AUROC. BCE receives logits; AUROC receives probabilities. Checkpoint selection maximizes validation only.'),nbf.v4.new_code_cell(trainer),nbf.v4.new_markdown_cell('### PROVIDED · historical task recovery'),nbf.v4.new_code_cell((P/'relkit/historical_task_l128.py').read_text()),nbf.v4.new_markdown_cell('### PROVIDED · every-batch audit'),nbf.v4.new_code_cell((P/'relkit/batch_audit_l123.py').read_text())])
 for name in ['resnet.py','sage_conv.py','hetero_conv.py','stype_encoder.py']:
  cells.append(nbf.v4.new_markdown_cell('### Read-only primitive · '+name+'\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full five-seed training · explicit gate\n\nOFF by default. Requires exact GPU environment in the reproduction contract, native pyg-lib and pinned text model. Run in a fresh directory. This gate has no monetary enforcement; the Modal runner enforces the author reservation budget. Live Colab NOT_CHECKED.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit the three-task contract table, three implementations and five written defenses from the lesson. Passing author/reference execution does not establish mastery. PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l128-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
reference='''## Task contracts at a glance

| Output family | Head | Released example loss | Selection | Time |
|---|---|---|---|---|
| Binary entity | One logit; sigmoid for evaluation | BCEWithLogits | AUROC, maximize | Query cutoff and mature future window |
| Regression entity | One scalar | L1 | MAE, minimize | Query cutoff and mature future window |
| Recommendation | Candidate scores → top-k IDs | BPR (two-tower example) | MAP@k, maximize | Query cutoff and eligible candidates |

## Metric definitions

AUROC = mean positive-negative ordering credit, with 0.5 for ties. Requires both classes. MAE = mean absolute difference in target units. AP@k = sum of precision at relevant ranks divided by min(k, number relevant). MAP averages eligible queries; pinned RelBench excludes zero-positive queries. Reject duplicate candidate IDs. A hit rate is a different metric.

## Different axes

Entity/recommendation describes the output. Temporal describes information availability and splits. Autoregressive describes conditioning later outputs on earlier outputs; it is not a TaskType in the pinned API. Teacher forcing supplies true earlier outputs during training, while generation may supply model outputs.

## DNF provenance trap

Paper Table13 has historical positive counts1365/125/207. Current archive has10046/441/495. Historical positive means no statusId !=1 in the future window; the later task flips it. Every recovered key/label matches old SQL, but historical archive bytes/order remain unestablished. Reversing both labels and probabilities preserves AUROC, not task semantics.

[Lesson](../lessons/0128-task-taxonomy.html) · [Protocol and commands](../labs/l128-reproduction.md) · [Evidence](../labs/evidence/l128/summary.json) · [Paper](https://arxiv.org/html/2407.20060v1) · [Pinned metrics](../labs/sources/l128/metrics.py).
'''
(R/'reference/task-taxonomy.html').write_text(document('Task taxonomy · reference',reference))
print('Built Lesson128, reference and portable notebooks')

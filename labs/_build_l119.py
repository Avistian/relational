"""Build the synthesis lesson and portable, independently executable notebooks."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0119-year-3-synthesis';TITLE='Year 3 synthesis: graphs versus flat tables'
CANONICAL=(P/'relkit/synthesis_l119.py').read_text()
CAP={
 'collision':('l119','A chosen four-number summary erases temporal order. An engineered time-weighted sum or suitable message sum restores this distinction: 220 versus 140.'),
 'expressiveness':('l119','Equal initial states and two equal neighbors per node stay equal in a six-cycle and two triangles. The displayed shared sum update cannot detect connectivity.'),
 'architecture':('l117','The reused released RelBench regression model: query-time sampling, per-table encoders, two typed GraphSAGE layers, seed readout, loss and validation selection.'),
 'results':('l119','Fresh L119 seeds and published RDL means. The diamond and bar show the fresh mean and sample seed standard deviation, not a confidence interval or a matched baseline comparison.')}
TASKS={
 'collision_ceiling':('check_ceiling','Find identical-feature groups and count their majority labels.','This detects information lost before any classifier trains.'),
 'typed_sum':('check_typed','Send weighted source values to their receiving nodes using each relation type.','The same values can mean different things under different relationships.'),
 'eligible_nodes':('check_time','Expand legal neighbors with one fixed query cutoff and both clocks.','A late-arriving event is not available just because its event date is old.'),
 'aligned_mae':('check_mae','Join predictions and targets by complete, unique query identity before scoring.','Entity IDs repeat; row order is not a reliable join key.')}
CHECK_SOURCE=(P/'_check_l119.py').read_text()
CHECKS={n.name:ast.get_source_segment(CHECK_SOURCE,n) for n in ast.parse(CHECK_SOURCE).body if isinstance(n,ast.FunctionDef)}

def results():
 s=json.loads((P/'evidence/l119/summary.json').read_text())
 txt='**Author-reference evidence: five fresh full-data fits for Lesson 119.** The notebook default only rescores these stored predictions.\n\n| Split | Fresh mean MAE | Sample seed SD | Published RDL mean | Verdict |\n|---|---:|---:|---:|---|\n'
 for split in ['val','test']:
  m=s['metrics'][split];txt+=f"| {split} | {m['mean']:.5f} | {m['sample_sd']:.5f} | {m['target']:.3f} | {m['verdict']} |\n"
 txt+='\n| Seed | Selected epoch | Validation MAE | Test MAE |\n|---|---:|---:|---:|\n'
 for row in s['seeds']:txt+=f"| {row['seed']} | {row['selected_epoch']} | {row['val']:.5f} | {row['test']:.5f} |\n"
 txt+=f"\nAll **{s['evaluated_queries']:,}** final query predictions were independently rescored. The largest raw-output discrepancy from the separately loaded original model on the same inputs was **{s['maximum_original_output_error']:.3g}**. Selected experiment: **COMPLETE**; whole-paper parity: **NOT_ESTABLISHED**.\n\n"
 txt+=f"Pilot plus five-run measured worker resource estimate: **USD{s['worker_resource_usd']:.4f}**. This excludes unitemized build/startup/storage charges. Reservations cap eight worker-hours at USD6.500736, with USD3.499264 held for overhead under the USD10 plan. [Fresh evidence](../labs/evidence/l119/summary.json) · [Budget](../labs/_budget_l119.json).\n"
 return txt

def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for name,(folder,caption) in CAP.items():
  path=P/f'figures/{folder}/{name}.png'
  source='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode() if portable else f'../labs/figures/{folder}/{name}.svg'
  portable_style=' style="min-width:620px;width:100%;max-width:none"' if portable else ''
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0" style="max-width:100%;overflow:auto"><img src="{source}" alt="{caption}"{portable_style}><figcaption>{caption} Scroll sideways on narrow screens.</figcaption></figure>')
 for token,ident,fallback in [
  ('WARMUP','warmup','Without notes: explain one aggregation collision, one limit of message passing, and the timestamp that governs a two-hop prediction.'),
  ('COLLISION_WIDGET','collision','Static trace: [3,90,30,50] for both rows gives a 50% observed ceiling. Add time-weighted sums 220 and 140: the ceiling becomes 100% on these two examples.'),
  ('MESSAGE_WIDGET','messages','Static trace: C=8 gives A=3 after one round and A=3.75 after two. C=20 keeps A=3 after one round and changes its two-round state to 5.25.'),
  ('TEACHBACK','teachback','Explain why both a lossy flat summary and a limited graph model can fail. State what the fresh selected reproduction adds, and which baseline experiment is still missing.')]:
  text=text.replace('[['+token+']]',fallback if portable else f'<div id="l119-{ident}"></div><noscript><p>{fallback}</p></noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/')
  text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 styles=''.join(f'<link rel="stylesheet" href="../assets/{name}.css">' for name in ['lesson','event-snapshot','reproduction','message-passing-viz'])
 scripts=''.join(f'<script src="../assets/{name}.js"></script>' for name in ['retrieval-pool','retrieval-bank','teachback','message-passing-viz','representation-collision-viz','l119-lesson']) if interactive else ''
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+styles+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0118-cvitkovic-relational-gnn.html">Lesson 118</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 119</p><h1>'+title+'</h1></header>'+html+'</article>'+scripts+'</body></html>'

bootstrap='''# @colab-bootstrap: install missing dependencies; report actual versions.
import importlib.util,importlib.metadata,subprocess,sys
required={'numpy':'numpy','pandas':'pandas','pyarrow':'pyarrow','torch':'torch','torch_frame':'pytorch-frame==0.2.3','torch_geometric':'torch-geometric==2.6.1','relbench':'relbench==1.1.0','pooch':'pooch'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Actual notebook runtime:',sys.version)
print({k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','pytorch-frame','torch-geometric','relbench']})
'''
payload={}
for seed in range(5):
 raw=(P/f'evidence/l119/paper/seed-{seed}/predictions.npz').read_bytes()
 payload[str(seed)]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
rescore='''# PROVIDED: actual fresh-run predictions, hash-checked before your scorer uses them.
import base64,hashlib,io,json
AUTHOR_PREDICTIONS = '''+repr(payload)+'''
rescored=[]
for seed,artifact in AUTHOR_PREDICTIONS.items():
    raw=base64.b64decode(artifact['base64'])
    assert hashlib.sha256(raw).hexdigest()==artifact['sha256']
    arrays=np.load(io.BytesIO(raw),allow_pickle=False)
    scores={}
    for split in ['val','test']:
        ids=[str(int(e))+'@'+str(int(t)) for e,t in zip(arrays[split+'_entity'],arrays[split+'_time'])]
        # Reverse prediction order deliberately. Target order is unchanged.
        scores[split]=aligned_mae(ids[::-1],arrays[split+'_pred'][::-1],ids,arrays[split+'_target'])
    rescored.append(dict(seed=int(seed),**scores))
print('Fresh AUTHOR runs rescored by your function; no training in this cell:')
for row in rescored:print(row)
expected = '''+repr(json.loads((P/'evidence/l119/summary.json').read_text())['seeds'])+'''
for actual,reference in zip(rescored,expected):
    for split in ['val','test']:assert abs(actual[split]-reference[split])<1e-12
course=course_experiment()
assert course['coarse_ceiling']==.5 and course['augmented_ceiling']==1.
assert course['indistinguishable'] and course['typed_outputs']==[6.,-6.]
assert course['eligible_nodes']==[0,1,3]
report=dict(status='PASS',course=course,rescored=rescored,query_predictions=6295,
            fresh_training_in_this_kernel=False,learner_status='PENDING_WRITTEN_DEFENSE')
from pathlib import Path
Path('l119-task-report.json').write_text(json.dumps(report,indent=2))
print('PASS: four live functions; 6,295 fresh-run query predictions rescored. Written defense pending.')
'''
old=nbf.read(P/'solutions/0117-rdl-bridge.ipynb',as_version=4)
gate=next(c.source for c in old.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source)
gate=gate.replace("Path('l117-full')","Path('l119-full')")
gate=gate.replace("if RUN_FULL_REPRODUCTION:\n", "if RUN_FULL_REPRODUCTION:\n    assert torch.cuda.is_available(), 'Use the documented compatible GPU environment'\n    import pyg_lib\n    for package,version in {'torch':'2.5.1','relbench':'1.1.0','pytorch-frame':'0.2.3','torch-geometric':'2.6.1','sentence-transformers':'3.3.1','numpy':'1.26.4','pandas':'2.2.3'}.items():\n        assert importlib.metadata.version(package).split('+')[0]==version, 'Use the pinned runtime: '+package\n")

(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 119 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Four live TODO/CHECK tasks and a one-page written synthesis. Default: exact course examples plus scoring of embedded real F1 predictions. Full training is a separate OFF-by-default gate. Prediction artifacts are author evidence, not your own model training.'),nbf.v4.new_code_cell(bootstrap)]
 cells.append(nbf.v4.new_markdown_cell('## Execution contract\n\nThe default notebook needs no checkout, private path or external dataset download. Figures and fresh prediction arrays are embedded. It executes complete model definitions but does not train by default. The bootstrap installs only missing packages and reports your actual versions; it does not certify a historical environment. The optional full reproduction requires the pinned Python3.11/CUDA12.4 runtime and native pyg-lib described in the contract. Live Colab NOT_CHECKED.\n\n**Data tiers:** the authored pair and graph examples isolate mechanisms; the embedded 6,295 predictions come from five fresh full-data RelBench runs. Neither kind of evidence substitutes for the other.'))
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Implement the four evidence checks\n\nComplete each TODO before running its CHECK. The final course and real-prediction cells call these exact functions. Checks give feedback on rejected cases; passing them does not grade the written argument.'))
 for chunk in re.split(r'^# %% ',CANONICAL,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  if task:
   check,goal,why=TASKS[task]
   cells.append(nbf.v4.new_markdown_cell('### '+heading+'\n\n**Goal:** '+goal+'\n\n**Why:** '+why+'\n\n**Hint boundary:** use the stated information contract; do not change the CHECK.'))
   if not solution:
    node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
    body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  else:cells.append(nbf.v4.new_markdown_cell('### '+heading))
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.append(nbf.v4.new_code_cell(CHECKS[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · exact mechanisms and real prediction audit\n\nPredict the representation ceilings and graph states before running. Your identity-aware scorer then joins intentionally reversed predictions to the correct real query targets. The arrays are the Lesson119 author runs, not a new fit in this notebook.'),nbf.v4.new_code_cell(rescore)])
 cells.append(nbf.v4.new_markdown_cell('## Inspect the complete selected reproduction\n\nThe following definitions are the unchanged L117 released-protocol implementation used in the NEW L119 runs. It includes graph construction, query transforms, encoders, heterogeneous messages, optimizer, complete epoch loop, validation selection and original-model replay. The upstream RelBench portions retain their MIT provenance; [source and licenses](https://avistian.github.io/relational/labs/sources/l117/manifest.json). These PROVIDED definitions do not supply answers to the four synthesis TODOs.'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1)
  cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for name in ['resnet.py','sage_conv.py']:
  cells.append(nbf.v4.new_markdown_cell('### Pinned library operator · '+name+'\n\nThe model calls the installed library implementation. The source is shown for inspection, not executed as a partial replacement.\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## NEXT STEP · full five-seed reproduction, default OFF\n\nThe cell uses the visible model above, downloads hash-checked full data, and trains all five seeds for ten epochs. Use the exact runtime in the [reproduction contract](https://avistian.github.io/relational/labs/l119-reproduction.md). The local notebook gate does not enforce a monetary ceiling; the supplied Modal operator reserves runtime under the USD10 aggregate cap. The executed solution leaves this gate OFF; separate author evidence records the completed GPU runs.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT · code plus written defense\n\nSubmit your four functions, `l119-task-report.json`, and a 500–700-word synthesis following Section8. Score 0–2 for the conditional claim, mechanism, counterargument, evidence discipline and falsifiable test. Include the absence of a fresh matched tabular baseline and L118 NOT_RUN. **PENDING_WRITTEN_DEFENSE**. Revisit the argument in 1, 7 and 30 days.')])
 for i,c in enumerate(cells):c.id=f'l119-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  previous=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in previous.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=previous.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)

reference='''## Three different questions

**Information:** what distinctions survive extraction and encoding? **Computation:** can the chosen operators use those distinctions? **Evidence:** does the trained predictor improve under a fair protocol?

## Exact course witnesses

- Ada [10,30,50] and Bo [50,30,10], at times [1,2,3], both map to [3,90,30,50]. On authored rising/falling labels the observed deterministic ceiling is .5. Sum(time×amount) gives220/140 and separates the two rows. A richer flat feature can repair the collision.
- Six-cycle versus two triangles: six equal initial states, degree2, shared local sum update. States1→3→9→27→81 in both. More rounds do not break this symmetry.
- Purchase/refund weights+1/−1 give+6/−6 for values10/4 with swapped roles. Untyped sum gives14/14.
- Query cutoff7 excludes event5/arrival9 and event11/arrival11; event6/arrival6 is legal. Preserve the root cutoff through timeless dimensions.

## Audit rules

Observed collision ceilings are not population performance estimates. Query identity includes entity and time. Match the full query population before scoring. Seed SD is not a cross-dataset uncertainty estimate. A fresh RDL score versus a historical LightGBM number is not a paired experiment.

## Selected reproduction

Five fresh full-data10-epoch RelBench v1 Table7 F1 RDL runs. Complete selected released-protocol execution; historical identity and whole-paper parity NOT_ESTABLISHED. Source fanout[128,64] differs from paper table128. Feature statistics are test-censored-database fits, not train-only. Real ingestion histories unavailable. L118 Home Credit remains NOT_RUN.

## One-page argument

Conditional claim → mechanism and flat-feature repair → graph counterexample → cited/measured evidence and deviations → predefined falsifying comparison. Rubric0–2 per part; PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0119-year-3-synthesis.html) · [Template](../labs/l119-writing-template.md) · [Protocol](../labs/l119-reproduction.md) · [Fresh results](../labs/evidence/l119/summary.json).
'''
(R/'reference/year-3-synthesis.html').write_text(document('Graphs versus flat tables · quick reference',reference))
print('Built L119 lesson, reference, and student/solution notebooks')

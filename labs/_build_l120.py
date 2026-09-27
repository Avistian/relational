"""Build exam, reference, and standalone notebooks from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0120-year-3-exit-exam';TITLE='Year 3 exit exam: a heterogeneous temporal GNN'
canonical=(P/'relkit/exam_l120.py').read_text();check_source=(P/'_check_l120.py').read_text()
checks={n.name:ast.get_source_segment(check_source,n) for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef)}
tasks={'key_edges':'check_keys','visible_rows':'check_visibility','seed_positions':'check_roots','seed_loss':'check_loss','typed_messages':'check_messages'}
summary=json.loads((P/'evidence/l120/summary.json').read_text())
results='**Fresh author-reference execution: COMPLETE selected released-protocol experiment.**\n\n| Split | Mean MAE | Sample seed SD | Published mean | Verdict |\n|---|---:|---:|---:|---|\n'
for split in ['val','test']:
 m=summary['metrics'][split];results+=f"| {split} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+='\n| Seed | Selected epoch | Validation MAE | Test MAE |\n|---|---:|---:|---:|\n'
for row in summary['seeds']:results+=f"| {row['seed']} | {row['selected_epoch']} | {row['val']:.6f} | {row['test']:.6f} |\n"
results+=f"\nAll **{summary['evaluated_queries']:,}** final predictions were independently scored. Maximum original-model output discrepancy: **{summary['maximum_original_output_error']:.3g}**. Measured pilot plus full-fit worker resource estimate: **USD {summary['worker_resource_usd']:.4f}**, excluding unitemized overhead. The aggregate plan caps reservations plus overhead reserve at USD 10. [Evidence](../labs/evidence/l120/summary.json).\n"
captions={'cutoff':'A day-7 person90 query admits event1; late event2 enters only at day8. Both clocks stay bounded at every hop.', 'architecture':'The course model: typed two-channel inputs, table encoders, two relation-specific message rounds, query head and mature-label seed loss.', 'paper':'The separate published RelBench model: table encoders, query-relative time, typed GraphSAGE and seed regression. Reused L117 architecture, executed afresh for L120.'}
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l120/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l120/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0" style="max-width:100%;overflow:auto"><img src="{src}" alt="{caption}"{(' style="min-width:620px;width:100%;max-width:none"' if portable else '')}><figcaption>{caption}</figcaption></figure>')
 fallback='Static trace: at day7, person90 sees event1. At day8, event2 arrives. Event5 first becomes visible at day11.'
 text=text.replace('[[CUTOFF_WIDGET]]',fallback if portable else '<div id="l120-cutoff"></div><noscript><p>'+fallback+'</p></noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0119-year-3-synthesis.html">Lesson 119</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 120</p><h1>'+title+'</h1></header>'+body+'</article>'+('<script src="../assets/l120-lesson.js"></script>' if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0119-year-3-synthesis.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l119','l120')
payload={str(seed):{'base64':base64.b64encode((P/f'evidence/l120/paper/seed-{seed}/predictions.npz').read_bytes()).decode(),'sha256':hashlib.sha256((P/f'evidence/l120/paper/seed-{seed}/predictions.npz').read_bytes()).hexdigest()} for seed in range(5)}
rescore='''# PROVIDED: independently rescore all fresh author-reference queries.
import numpy as np,io,base64,hashlib
payload = '''+repr(payload)+'''
expected = '''+repr(summary['seeds'])+'''
for seed,item in payload.items():
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    arrays=np.load(io.BytesIO(raw),allow_pickle=False)
    for split in ['val','test']:
        ids=list(zip(arrays[split+'_entity'].tolist(),arrays[split+'_time'].tolist()))
        assert len(set(ids))==len(ids)
        prediction=dict(zip(ids[::-1],arrays[split+'_pred'][::-1].tolist()))
        mae=sum(abs(prediction[key]-float(y)) for key,y in zip(ids,arrays[split+'_target']))/len(ids)
        assert abs(mae-expected[int(seed)][split])<1e-12
        print(seed,split,mae)
print('6,295 fresh author query predictions rescored; this cell does not train.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 120 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student exam')+' · Five live functions feed a real PyG training pipeline. Default: small authored training plus rescoring embedded full-data author predictions. Full GPU training remains a separate gate. Learner status PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((name for name in tasks if 'def '+name+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\nImplement this function, then run its unchanged CHECK. The complete training pipeline calls your function.' if task else '')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.append(nbf.v4.new_code_cell(checks[tasks[task]]+'\n'+tasks[task]+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · your complete small PyG pipeline\n\nPredict the legal event IDs and direct loss mask before execution. The report proves reference execution, not exam mastery.'),nbf.v4.new_code_cell("report=course_run()\nfrom pathlib import Path\nPath('l120-task-report.json').write_text(json.dumps(report,indent=2))\nprint(json.dumps(report,indent=2))"),nbf.v4.new_markdown_cell('## Audit the real author runs\n\nThe embedded predictions are from five fresh full-data fits. They do not replace your own training evidence.'),nbf.v4.new_code_cell(rescore)])
 cells.append(nbf.v4.new_markdown_cell('## Full published implementation and trainer · PROVIDED\n\nUnchanged pinned RelBench release model/trainer from L117, freshly executed in L120. This differs from the small course model. Source and MIT licenses: [manifest](https://avistian.github.io/relational/labs/sources/l117/manifest.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for name in ['resnet.py','sage_conv.py']:
  cells.append(nbf.v4.new_markdown_cell('### Released primitive · '+name+'\n\nRead-only source of the installed library operator, shown for inspection.\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full five-seed reproduction · OFF by default\n\nRequires the exact pinned GPU runtime in the contract, including native pyg-lib. The notebook gate does not enforce a monetary cap; the supplied Modal runner does reserve bounded compute. No historical environment or live Colab claim is made.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit your graph, implementations, executed artifacts, evidence labels and completed [Fey reading/defense ledger](https://avistian.github.io/relational/labs/l120-submission.md). Passing checks alone leaves PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l120-{i:03d}'
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
reference='''## Six contracts to defend

| Boundary | Required invariant | Failure witness |
|---|---|---|
| Table → graph | Keys map to row positions; typed reverse edges are explicit | key90 versus row0 |
| History → query | Every hop obeys root event and availability cutoff | event5 / arrival8 at query7 |
| Graphs → batch | Queries remain disjoint; locate actual roots | roots2/3 versus prefix0/1 |
| Batch → loss | Only seed outputs supervised; labels mature at fit time | direct context gradient or arrival after fit |
| Fit → score | Validation chooses checkpoint; test follows | test-selected epoch |
| Evidence → claim | Source/data/protocol and actual execution identified | course loss presented as paper MAE |

## Course trace

person90 at day7, two hops: person90 → event1 → merchant8. At day8, event2 and merchant4 become legal. Input vectors2 → table vectors8 → two typed message rounds → scalar query head. Original teaching network, not the published model.

## Full experiment

Five fresh10-epoch F1 RDL fits; all7,453/499/760 train/validation/test queries. Complete selected released-protocol execution, not whole-paper parity. Fanout and preprocessing deviations remain recorded; real ingestion histories unavailable.

## Exit artifacts

Hand-built REG, five functions, working PyG pipeline, leakage counterexample, source/result audit, full Fey reading ledger and written defense. Six domains0–2; at least10/12, no zero, all critical correctness gates and full reading required. PENDING_WRITTEN_DEFENSE until demonstrated.

[Lesson](../lessons/0120-year-3-exit-exam.html) · [Submission](../labs/l120-submission.md) · [Protocol](../labs/l120-reproduction.md).
'''
(R/'reference/year-3-exit-exam.html').write_text(document('Year 3 exit · reference',reference))
print('Built L120 lesson, reference and portable notebooks')

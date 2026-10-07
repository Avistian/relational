"""Build lesson, reference, and standalone notebooks from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0124-entity-task-tables';TITLE='Entity vs task table: what question does this row ask?'
canonical=(P/'relkit/tasks_l124.py').read_text();check_source=(P/'_check_l124.py').read_text()
checks={n.name:ast.get_source_segment(check_source,n) for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef)}
task_notes={'make_labels':'Construct future means with explicit released/past cohort choice. Hint: separate eligibility, window bounds and aggregation.', 'validate_task':'Check query identity, entity references and label maturity. Hint: entity repetition is legal; query repetition is not.', 'aligned_mae':'Score by query keys after deliberately shuffling predictions. Hint: require a one-to-one outer join with complete coverage.'}
tasks={'make_labels':'check_labels','validate_task':'check_validate','aligned_mae':'check_alignment'}
summary=json.loads((P/'evidence/l124/summary.json').read_text())
results='**Fresh author-reference execution: COMPLETE selected released-protocol experiment.**\n\n| Split | Mean MAE | Sample seed SD | Published mean | Verdict |\n|---|---:|---:|---:|---|\n'
for split in ['val','test']:
 m=summary['metrics'][split];results+=f"| {split} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+='\n| Seed | Selected epoch | Validation MAE | Test MAE |\n|---|---:|---:|---:|\n'
for row in summary['seeds']:results+=f"| {row['seed']} | {row['selected_epoch']} | {row['val']:.6f} | {row['test']:.6f} |\n"
results+=f"\nAll **{summary['evaluated_queries']:,}** final predictions were independently scored. Maximum original-model output discrepancy: **{summary['maximum_original_output_error']:.3g}**. Measured pilot plus full-fit worker resource estimate: **USD {summary['worker_resource_usd']:.4f}**, excluding unitemized overhead. The aggregate plan caps reservations plus overhead reserve at USD 10. [Evidence](../labs/evidence/l124/summary.json).\n"
task_report=json.loads((P/'_task_audit_l124_results.json').read_text())
task_results='**Complete task reconstruction: PASS — all 8,712 rows.**\n\n| Split | Task rows | Grid cutoffs | Nonempty cutoffs | Rows excluded by past-only activity |\n|---|---:|---:|---:|---:|\n'
for split,r in task_report['splits'].items():task_results+=f"| {split} | {r['rows']} | {r['timestamps']} | {r['nonempty_timestamps']} | {r['released_only_rows']} |\n"
task_results+='\n[Independent task audit](../labs/_task_audit_l124_results.json). Past-only counts are a course intervention, not a changed benchmark score.'
results+=f"\nAll sampled batches also passed the inherited root-time/edge/query-isolation audit: **{summary['temporal_audit']['audited_query_occurrences']:,} query occurrences**, including repeated epochs. Availability and historical feature values remain unobserved.\n"
captions={'identity':'One driver node supports separate task rows at separate cutoffs. The task row selects the seed and supervision; its label is not a graph feature.', 'forward':'Task input drives temporal sampling, typed width-128 encoding, two GraphSAGE layers and a scalar seed head. Future labels enter only the loss branch.', 'scores':'Five fresh complete fits; diamonds and bars show mean plus or minus sample seed SD. Dashed lines are published targets. Detail scales differ.'}
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[TASK_RESULTS]]',task_results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l124/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l124/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0" style="max-width:100%;overflow:auto"><img src="{src}" alt="{caption}"{(' style="min-width:620px;width:100%;max-width:none"' if portable else '')}><figcaption>{caption}</figcaption></figure>')
 fallback='Static trace: at day0, past activity admits Driver0 with future positions2 and6: mean4. Released-style eligibility also admits Driver1 with future position3. No future event means no observed mean, not zero.'
 text=text.replace('[[TASK_WIDGET]]',fallback if portable else '<div id="l124-task"></div><noscript><p>'+fallback+'</p></noscript>')
 text=text.replace('[[WARMUP]]','Recall without notes: which rows can a prediction at time t see, and what does a foreign key identify?' if portable else '<div id="warmup"></div><noscript><p>Recall: which rows can a prediction at time t see?</p></noscript>')
 text=text.replace('[[TEACHBACK]]','Explain the difference between future label evidence and information available for a prediction.' if portable else '<div id="l124-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','task-table'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0123-temporal-heterogeneous-graphs.html">Lesson 123</a></nav><header><p class="stream-kicker">Year 4 · Quarter 1 · Lesson 124</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','task-table-viz','l124-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0119-year-3-synthesis.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l119','l124')
gate=gate.replace("    records=[fit_rdl", """    OriginalLoader=NeighborLoader
    class NotebookAuditedLoader(OriginalLoader):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.original_graph=args[0];self.entity=kwargs['input_nodes'][0]
        def __iter__(self):
            for batch in super().__iter__():
                audit_batch(batch,self.original_graph,self.entity)
                yield batch
    NeighborLoader=NotebookAuditedLoader
    records=[fit_rdl""")
gate=gate.replace("    print({s:float", "    NeighborLoader=OriginalLoader\n    print({s:float")
payload={str(seed):{'base64':base64.b64encode((P/f'evidence/l124/paper/seed-{seed}/predictions.npz').read_bytes()).decode(),'sha256':hashlib.sha256((P/f'evidence/l124/paper/seed-{seed}/predictions.npz').read_bytes()).hexdigest()} for seed in range(5)}
rescore='''# PROVIDED: your query-key scorer consumes all fresh author predictions.
import numpy as np,io,base64,hashlib
payload = '''+repr(payload)+'''
expected = '''+repr(summary['seeds'])+'''
for seed,item in payload.items():
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    arrays=np.load(io.BytesIO(raw),allow_pickle=False)
    for split in ['val','test']:
        rows=pd.DataFrame({'driverId':arrays[split+'_entity'], 'date':pd.to_datetime(arrays[split+'_time']), 'position':arrays[split+'_target']})
        predictions=rows[['driverId','date']].copy();predictions['prediction']=arrays[split+'_pred'];predictions=predictions.sample(frac=1,random_state=124)
        mae=aligned_mae(rows,predictions)
        assert abs(mae-expected[int(seed)][split])<1e-12
        print(seed,split,mae)
print('6,295 fresh author predictions rescored by your function; this cell does not train.')
'''
task_bytes=(P/'evidence/l124/task-inputs.json.gz').read_bytes()
full_task_code="""# PROVIDED harness: all events and task rows, called through your functions.
import gzip,base64,hashlib,numpy as np
raw=base64.b64decode(TASK_PAYLOAD);assert hashlib.sha256(raw).hexdigest()==TASK_HASH
real=json.loads(gzip.decompress(raw));events=pd.DataFrame(real['events']);events['date']=pd.to_datetime(events.date)
all_rows={}
for split,spec in real['splits'].items():
    source=events if spec['db_cutoff'] is None else events[events.date<=pd.Timestamp(spec['db_cutoff'])]
    rows=make_labels(source,pd.to_datetime(spec['cutoffs']))
    rows=rows[rows.driverId.isin(real['entity_ids'])]
    validate_task(rows,real['entity_ids'],real['val_timestamp'] if split=='train' else None)
    expected=pd.DataFrame(spec['expected']);expected['date']=pd.to_datetime(expected.date)
    a=rows.sort_values(['date','driverId']).reset_index(drop=True);b=expected.sort_values(['date','driverId']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[['driverId','date']],b[['driverId','date']],check_dtype=False)
    np.testing.assert_allclose(a.position,b.position,rtol=0,atol=1e-12)
    past=make_labels(source,pd.to_datetime(spec['cutoffs']),'past')
    print(split,len(rows),'rows exactly reconstructed;',len(rows)-len(past),'excluded by past activity')
    all_rows[split]=rows
assert sum(map(len,all_rows.values()))==8712
""".replace('TASK_PAYLOAD',repr(base64.b64encode(task_bytes).decode())).replace('TASK_HASH',repr(hashlib.sha256(task_bytes).hexdigest()))
graph_bytes=(P/'evidence/l123/temporal-graph.json.gz').read_bytes()
# L123's complete topology is inherited, with its checksum and oracle identities.
full_graph_code="""# PROVIDED: trace one REAL task row to its exact input subgraph.
raw=base64.b64decode(GRAPH_PAYLOAD);assert hashlib.sha256(raw).hexdigest()==GRAPH_HASH
graph=json.loads(gzip.decompress(raw));q=graph['queries'][0]
nodes={(kind,i):None if kind not in graph['times'] else (graph['times'][kind][i],graph['times'][kind][i]) for kind,n in graph['rows'].items() for i in range(n)}
edges=[]
for key,(source,dest) in graph['edges'].items():
    src,rel,dst=key.split('|')
    edges.extend(dict(src=(src,a),dst=(dst,b),kind=rel,stamp=None) for a,b in zip(source,dest))
sample=sample_temporal(nodes,edges,('drivers',q['entity']),q['cutoff'],2);audit_sample(nodes,edges,sample)
oracle=graph['query_oracles'][q['split']+':'+str(q['entity'])+':'+str(q['cutoff'])]
assert sorted([list(n) for n in sample['nodes']])==oracle['nodes'] and sample['edges']==oracle['edges']
rows=all_rows[q['split']];row=rows[(rows.driverId==q['entity'])&(rows.date==pd.to_datetime(q['cutoff'],unit='s'))]
assert len(row)==1
annotation={'task_row':json.loads(row.to_json(orient='records',date_format='iso'))[0],'input_nodes':[list(n) for n in sample['nodes']],'input_edge_indices':sample['edges'],'label_is_input':False,'seed_shape':[1,128],'prediction_shape':[1,1]}
Path('l124-example-subgraph.json').write_text(json.dumps(annotation,indent=2))
print(annotation['task_row']);print(len(sample['nodes']),'nodes;',len(sample['edges']),'edges; one supervised prediction')
""".replace('GRAPH_PAYLOAD',repr(base64.b64encode(graph_bytes).decode())).replace('GRAPH_HASH',repr(hashlib.sha256(graph_bytes).hexdigest()))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 124 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three live functions construct labels, validate task contracts and align predictions. PROVIDED = inspect; TODO = implement; CHECK = execute unchanged; EXIT = defend. Default execution reconstructs all 8,712 task rows, traces a real input subgraph and rescores author predictions; full GPU training is separately gated. Learner status PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((name for name in tasks if 'def '+name+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n'+task_notes[task]+' Implement this function, then run its unchanged CHECK. The toy trace and full F1 task harness call your function.' if task else '')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.append(nbf.v4.new_code_cell(checks[tasks[task]]+'\n'+tasks[task]+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · your task-table contract\n\nPredict which driver receives a label before execution. The report proves reference execution, not learner mastery.'),nbf.v4.new_code_cell("report=course_run()\nfrom pathlib import Path\nPath('l124-task-report.json').write_text(json.dumps(report,indent=2))\nprint(json.dumps(report,indent=2))"),nbf.v4.new_markdown_cell('## Audit the real author runs\n\nThe embedded predictions are from five fresh full-data fits. They do not replace your own training evidence.'),nbf.v4.new_code_cell(rescore)])
 cells.extend([nbf.v4.new_markdown_cell('## Reconstruct all released task rows\n\nYour functions process the complete result events and all split cutoffs. Source SQL was checked separately in the author audit.'),nbf.v4.new_code_cell(full_task_code)])
 temporal=(P/'relkit/temporal_l123.py').read_text()
 # Keep the inherited sampler without its unrelated toy entrypoint.
 node=next(n for n in ast.parse(temporal).body if isinstance(n,ast.FunctionDef) and n.name=='course_run')
 lines=temporal.splitlines();temporal='\n'.join(lines[:node.lineno-1]+lines[node.end_lineno:])
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · temporal sampler from L123\n\nReused visibly to construct the input for one real task row.'),nbf.v4.new_code_cell(temporal),nbf.v4.new_code_cell(full_graph_code)])
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · every-batch release audit\n\nThis checks identity and root-specific visibility; it cannot observe ingestion histories.'),nbf.v4.new_code_cell((P/'relkit/batch_audit_l123.py').read_text())])
 cells.append(nbf.v4.new_markdown_cell('## Full published implementation and trainer · PROVIDED\n\nUnchanged pinned RelBench release model/trainer from L117, freshly executed in L124. The teaching sampler is exhaustive; the reproduction loader uses bounded uniform fanouts. Source and MIT licenses: [manifest](https://avistian.github.io/relational/labs/sources/l117/manifest.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for name in ['resnet.py','sage_conv.py']:
  cells.append(nbf.v4.new_markdown_cell('### Released primitive · '+name+'\n\nRead-only source of the installed library operator, shown for inspection.\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full five-seed reproduction · OFF by default\n\nRequires the exact pinned GPU runtime in the contract, including native pyg-lib. The notebook gate does not enforce a monetary cap; the supplied Modal runner does reserve bounded compute. No historical environment or live Colab claim is made.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit your task table, implementations, executed artifacts, evidence labels and five written answers from the lesson. Passing checks alone leaves PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l124-{i:03d}'
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
reference="""## Task contract

| Field | Meaning / check |
|---|---|
| Entity | Foreign key to the prediction node; known entity |
| Cutoff | Information boundary for this query at every hop |
| Query key | (entity, cutoff) within a fixed task/horizon |
| Label window | F1: t < event time <= t + 60 days |
| Label maturity | Full window observed; add reporting delay if needed |
| Cohort | Declare whether eligibility is known at cutoff |
| No future event | No observed F1 mean; do not invent zero |
| Train loss | Seed predictions only; future labels are not features |
| Evaluation | One-to-one query-key match, finite predictions, complete coverage |

## Source caveat

Released F1 activity SQL has no upper date bound. A past-only bound changes the observed labeled cohort by955 train/33 validation/42 test rows. Even that intervention retains future-participation conditioning. No score for a changed task is claimed.

## Evidence

All8,712 task rows reconstructed; original SQL and archive labels agree. Five fresh full-data training runs are separate score evidence. Historical/whole-paper parity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0124-entity-task-tables.html) · [Protocol](../labs/l124-reproduction.md) · [Task audit](../labs/_task_audit_l124_results.json).
"""
(R/'reference/entity-task-tables.html').write_text(document('Entity vs task table · reference',reference))
print('Built L124 lesson, reference and portable notebooks')

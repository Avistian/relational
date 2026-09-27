"""Build aligned lesson/reference and portable notebooks from canonical code and evidence."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0127-relbench-v1';TITLE='RelBench v1: run and audit the RDL baseline'
canonical=(P/'relkit/benchmark_l127.py').read_text()
check_source=(P/'_check_l127.py').read_text();checks={n.name:ast.get_source_segment(check_source,n) for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef)}
functions={n.name:ast.get_source_segment(canonical,n) for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef)}
tasks={'experiment_contract':'check_contract','select_checkpoint':'check_selection','summarize_seeds':'check_summary'}
summary=json.loads((P/'evidence/l127/summary.json').read_text())
results='**Fresh author execution: COMPLETE selected released-protocol experiment.** Five new seeds, ten complete epochs each.\n\n| Split | Mean MAE | Sample seed SD | Paper mean | Verdict |\n|---|---:|---:|---:|---|\n'
for split in ['val','test']:
 m=summary['metrics'][split];results+=f"| {split} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+='\n| Seed | Selected epoch | Selection MAE | Final validation MAE | Test MAE |\n|---|---:|---:|---:|---:|\n'
for row in summary['seeds']:
 result=json.loads((P/f"evidence/l127/paper/seed-{row['seed']}/result.json").read_text())
 results+=f"| {row['seed']} | {row['selected_epoch']} | {result['selection_mae']:.6f} | {row['val']:.6f} | {row['test']:.6f} |\n"
results+=f"\nAll **{summary['evaluated_queries']:,}** final predictions independently rescored; **{summary['temporal_audit']['audited_query_occurrences']:,}** training/evaluation query occurrences audited. Maximum original-model output discrepancy **{summary['maximum_original_output_error']:.3g}**. Pilot plus five-fit worker resource estimate **USD {summary['worker_resource_usd']:.6f}**, excluding unitemized overhead; six bounded worker reservations consumed within the USD10 plan, no retries. [Evidence](../labs/evidence/l127/summary.json) · [Independent archive/SQL audit](../labs/_audit_l127_results.json).\n"
captions={'architecture':'Queries drive disjoint temporal sampling and table-specific Frame ResNets. Relative-time vectors, two sum-GraphSAGE layers and a scalar seed head connect to separate training and evaluation branches.', 'trace':'Real driver10 query at 2004-07-05: 99 nodes and 245 edges in the exhaustive neighborhood, one supervised output. Typed row counts determine encoder shapes; bounded training samples can differ.', 'selection':'Training, validation-only checkpoint selection, frozen test scoring and complete-seed aggregation are separate decisions.', 'scores':'Five fresh complete fits. Diamonds and error bars show mean and sample seed SD; dashed lines mark paper means. Each panel uses its own detail scale.'}
fallback='Static trace: validation MAE [3,2,2] selects epoch2, the first minimum. Test MAE [5,4,1] would select epoch3 if misused. Changing test scores must not change the valid selection.'
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 model_source=(P/'relkit/rdl_l117.py').read_text();model_tree=ast.parse(model_source)
 cls=next(n for n in model_tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
 forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 # Exact code shown beside its explanation; notebook definitions remain canonical.
 import textwrap
 fragment=textwrap.dedent(ast.get_source_segment(model_source,forward))
 # ast starts at def without indentation but subsequent lines retain class indentation.
 fragment='\n'.join([fragment.splitlines()[0]]+[line[4:] if line.startswith('    ') else line for line in fragment.splitlines()[1:]])
 text=text.replace('[[MODEL_CODE]]','```python\n'+fragment+'\n```')
 start=model_source.index('    for epoch in range(1,epochs+1):')
 end=model_source.index('    torch.save(state,',start)
 text=text.replace('[[TRAIN_CODE]]','```python\n'+textwrap.dedent(model_source[start:end])+'\n```')

 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l127/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l127/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 text=text.replace('[[CHECKPOINT_WIDGET]]',fallback if portable else '<div id="l127-checkpoint"></div><noscript><p>'+fallback+'</p></noscript>')
 text=text.replace('[[WARMUP]]','Recall the distinction between a context row, a supervised query and a checkpoint-selection metric.' if portable else '<div id="warmup"></div><noscript><p>Recall the distinction between context rows, queries and selection metrics.</p></noscript>')
 text=text.replace('[[TEACHBACK]]','Explain the strongest claim justified by five complete released-protocol fits and the evidence that supports it.' if portable else '<div id="l127-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','task-table'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0126-relbench-beta.html">Lesson 126</a></nav><header><p class="stream-kicker">Year 4 · Quarter 1 · Lesson 127</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','checkpoint-audit-viz','l127-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0124-entity-task-tables.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l124','l127')
gate=gate.replace("    records=[fit_rdl", "    frozen=experiment_contract()\n    records=[fit_rdl").replace("/f'seed-{seed}',10,", "/f'seed-{seed}',frozen['epochs'],").replace('for seed in range(5)]',"for seed in frozen['seeds']]")
payload={}
for seed in range(5):
 root=P/f'evidence/l127/paper/seed-{seed}';raw=(root/'predictions.npz').read_bytes()
 payload[str(seed)]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads((root/'result.json').read_text()),'completion':json.loads((root/'completed.json').read_text())}
real_harness='''# PROVIDED: real author evidence calls YOUR contract, selector and aggregator.
import io,base64,hashlib,json,math
from pathlib import Path
import numpy as np
payload = '''+repr(payload)+'''
expected = '''+repr(summary['metrics'])+'''
contract=experiment_contract();records=[];count=0
for seed in contract['seeds']:
    item=payload[str(seed)];result=item['result'];done=item['completion']
    assert select_checkpoint(result['trace'],contract['epochs'])==result['selected_epoch']
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    arrays=np.load(io.BytesIO(raw),allow_pickle=False);scores={}
    for split,n in [('val',499),('test',760)]:
        p=arrays[split+'_pred'];y=arrays[split+'_target']
        keys=list(zip(arrays[split+'_entity'],arrays[split+'_time']))
        assert len(p)==len(y)==len(set(keys))==n and np.isfinite(p).all()
        mae=math.fsum(abs(float(a)-float(b)) for a,b in zip(p,y))/n
        assert abs(mae-result['scores'][split])<1e-12
        scores[split]=mae;count+=n
    records.append(dict(seed=seed,status=done['status'],run_uuid=done['run_uuid'],epochs=result['epochs'],**scores))
    print('Seed',seed,'selected',result['selected_epoch'],'selection MAE',result['selection_mae'],'final',scores)
aggregate=summarize_seeds(records,contract['seeds'])
for split in ['val','test']:
    for key in ['mean','sample_sd']:assert abs(aggregate[split][key]-expected[split][key])<1e-12
report=dict(status='PASS',predictions=count,selected_epochs=[payload[str(s)]['result']['selected_epoch'] for s in contract['seeds']],aggregate=aggregate,scope='Complete embedded author evidence replay; no new training in this cell')
Path('l127-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
'''
graph_raw=(P/'evidence/l123/temporal-graph.json.gz').read_bytes();example=json.loads((P/'evidence/l124/example-subgraph.json').read_text())
trace_harness='''# PROVIDED: reconstruct one real exhaustive query context from the full released graph.
import gzip,collections
raw=base64.b64decode('''+repr(base64.b64encode(graph_raw).decode())+''')
assert hashlib.sha256(raw).hexdigest()=='''+repr(hashlib.sha256(graph_raw).hexdigest())+'''
graph=json.loads(gzip.decompress(raw));q=graph['queries'][0]
nodes={(kind,i):None if kind not in graph['times'] else (graph['times'][kind][i],graph['times'][kind][i]) for kind,n in graph['rows'].items() for i in range(n)}
edges=[]
for key,(source,dest) in graph['edges'].items():
    src,rel,dst=key.split('|');edges.extend(dict(src=(src,a),dst=(dst,b),kind=rel,stamp=None) for a,b in zip(source,dest))
sample=sample_temporal(nodes,edges,('drivers',q['entity']),q['cutoff'],2);audit_sample(nodes,edges,sample)
example='''+repr(example)+'''
assert sample['nodes']=={tuple(n) for n in example['input_nodes']} and sample['edges']==example['input_edge_indices']
assert len(sample['nodes'])==99 and len(sample['edges'])==245
counts=dict(collections.Counter(n[0] for n in sample['nodes']))
report=dict(status='PASS',query=example['task_row'],nodes=99,edges=245,typed_rows=counts,encoder_shapes={k:[n,128] for k,n in counts.items()},seed_shape=[1,128],output_shape=[1,1],boundary='Structural shapes, not recorded neural activations. Event time duplicated as availability; true arrival histories absent.')
Path('l127-trace.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 127 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three live functions own the experiment contract, checkpoint selection and seed aggregation. Default run replays complete fresh author artifacts and reconstructs a real input graph. PROVIDED = inspect; TODO = implement; CHECK = execute unchanged; EXIT = defend. Learner PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.extend([nbf.v4.new_markdown_cell('## Implement the experiment ledger\n\nEach function is used below on real artifacts. Hints: return fresh lists; minimum selection is stable on ties; check identities before averaging.'),nbf.v4.new_code_cell('import math,statistics,copy,json\n'+checks['rejected'])])
 for name,check in tasks.items():
  source=functions[name] if solution else functions[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
  cells.extend([nbf.v4.new_markdown_cell('### TODO · '+name),nbf.v4.new_code_cell(source),nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")')])
 cells.extend([nbf.v4.new_markdown_cell('## REPLAY · select checkpoints and rescore every prediction\n\nThese are author artifacts, not a new fit. Your functions determine membership, selection and aggregation.'),nbf.v4.new_code_cell(real_harness)])
 temporal=(P/'relkit/temporal_l123.py').read_text();tree=ast.parse(temporal)
 visible='import math\n\n'+'\n\n'.join(ast.get_source_segment(temporal,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['eligible','sample_temporal','audit_sample'])
 cells.extend([nbf.v4.new_markdown_cell('## TRACE · the complete graph and one real query\n\nVisible inherited exhaustive sampler from L123. The stochastic training loader is shown separately below.'),nbf.v4.new_code_cell(visible),nbf.v4.new_code_cell(trace_harness)])
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · every-batch temporal identity audit'),nbf.v4.new_code_cell((P/'relkit/batch_audit_l123.py').read_text())])
 cells.append(nbf.v4.new_markdown_cell('## Full released computation · visible source\n\nPinned upstream commit 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, MIT license. This code is reused from L117 and freshly executed in L127. Follow typed encoders → time → message passing → head → loss → checkpoint. [Source manifest and licenses](https://avistian.github.io/relational/labs/sources/l117/manifest.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for name in ['resnet.py','sage_conv.py','hetero_conv.py','stype_encoder.py']:
  cells.append(nbf.v4.new_markdown_cell('### Read-only library primitive · '+name+'\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full five-seed training · explicit gate\n\nDefault OFF. Requires the pinned GPU runtime, native pyg-lib and sentence-transformers from the reproduction contract. The gate checks core versions and archive hashes; it does not enforce a dollar cap. Use the bounded Modal checkout runner for the USD10 author plan. Live Colab NOT_CHECKED.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit your three implementations, artifacts and the five written defenses in the lesson. Rescoring author predictions and passing checks do not establish learner mastery. PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l127-{i:03d}'
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
reference='''## One experiment, five boundaries

| Stage | Fixed rule | Artifact to inspect |
|---|---|---|
| Pin | Same task, source, data, runtime and preprocessing | Hash manifest and deviation ledger |
| Train | Five fresh seeds; ten full epochs each | Epoch trace and query counts |
| Select | First minimum validation MAE | Selected epoch and weight hash |
| Score | Complete final queries from frozen checkpoint | IDs, timestamps, predictions, targets |
| Aggregate | Exact distinct completed seed set | Mean and sample SD (n−1 denominator) |

## Computation

Typed rows → per-table four-block ResNets → width128 row vectors + relative time → two typed sum-GraphSAGE layers → first B driver vectors → scalar head. Train with mean L1 over B queries. Evaluate with training-label percentile clipping. Every sampled dated node must obey its own root cutoff.

## Evidence language

Complete selected released-protocol replay is narrower than whole-paper reproduction. CLOSE means a predeclared descriptive mean tolerance, not statistical equivalence. Runtime/source/data identities and protocol gaps remain separate. Seed SD is not a confidence interval. A final validation resample can differ from the checkpoint-selection score.

## Protocol gaps to remember

Released fanouts [128,64] versus paper table128; released feature statistics through the test cap; historical seed/environment identity unavailable; true arrival and mutable-feature histories missing. Other29 tasks NOT_RUN here; historical/whole-paper parity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0127-relbench-v1.html) · [Exact commands and protocol](../labs/l127-reproduction.md) · [Measured evidence](../labs/evidence/l127/summary.json) · [Primary paper](https://arxiv.org/html/2407.20060v1).
'''
(R/'reference/relbench-v1.html').write_text(document('RelBench v1 · experiment reference',reference))
print('Built Lesson127, reference and portable student/solution notebooks')

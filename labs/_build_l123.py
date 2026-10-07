"""Build lesson, reference, and standalone notebooks from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0123-temporal-heterogeneous-graphs';TITLE='Temporal heterogeneous graphs: what may this query see?'
canonical=(P/'relkit/temporal_l123.py').read_text();check_source=(P/'_check_l123.py').read_text()
checks={n.name:ast.get_source_segment(check_source,n) for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef)}
task_notes={'eligible':'Goal: distinguish event time from arrival. Why: an old event may still be unavailable. Hint: validate both clocks and state the timeless assumption.', 'sample_temporal':'Goal: collect incoming typed context with one root cutoff. Why: hop-1 filtering alone leaks. Hint: maintain frontier, seen nodes and original edge identities.', 'audit_sample':'Goal: reject unknown, future or late rows/edges and missing endpoints. Why: graph counts do not prove visibility. Hint: an audit of visibility is not a proof of completeness.'}
tasks={'eligible':'check_eligible','sample_temporal':'check_sample','audit_sample':'check_audit'}
summary=json.loads((P/'evidence/l123/summary.json').read_text())
results='**Fresh author-reference execution: COMPLETE selected released-protocol experiment.**\n\n| Split | Mean MAE | Sample seed SD | Published mean | Verdict |\n|---|---:|---:|---:|---|\n'
for split in ['val','test']:
 m=summary['metrics'][split];results+=f"| {split} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+='\n| Seed | Selected epoch | Validation MAE | Test MAE |\n|---|---:|---:|---:|\n'
for row in summary['seeds']:results+=f"| {row['seed']} | {row['selected_epoch']} | {row['val']:.6f} | {row['test']:.6f} |\n"
results+=f"\nAll **{summary['evaluated_queries']:,}** final predictions were independently scored. Maximum original-model output discrepancy: **{summary['maximum_original_output_error']:.3g}**. Measured pilot plus full-fit worker resource estimate: **USD {summary['worker_resource_usd']:.4f}**, excluding unitemized overhead. The aggregate plan caps reservations plus overhead reserve at USD 10. [Evidence](../labs/evidence/l123/summary.json).\n"
graph_report=json.loads((P/'_temporal_audit_l123_results.json').read_text())
graph_results='**Real temporal audit: PASS.** Exact SQL node sets and PyG typed node/edge sets for nine specified queries.\n\n| Split | Driver row | Unix cutoff (seconds) | Nodes | Edges |\n|---|---:|---:|---:|---:|\n'
for q in graph_report['comparisons']:graph_results+=f"| {q['split']} | {q['entity']} | {q['cutoff']} | {q['nodes']} | {q['edges']} |\n"
graph_results+='\n[Full temporal report](../labs/_temporal_audit_l123_results.json). Equal-time, future-second-hop and mixed-query mutation checks are separate.'
results+=f"\nEvery training/evaluation batch passed timestamp, original-edge and query-isolation audits across **{summary['temporal_audit']['audited_query_occurrences']:,} query occurrences** (repeated across epochs). [Audit example](../labs/evidence/l123/paper/seed-0/temporal-audit.json).\n"
captions={'cutoff':'Synthetic day8 snapshot: Memo1 at day7 may inform Transfer0 at day4. Every hop uses root day8. Transfer2 occurred on day5 but arrived on day11.', 'paper':'Released RelBench RDL pipeline reused from L117: typed row encoders, query-relative time, two GraphSAGE rounds and seed prediction. L123 audits each sampled batch.', 'scores':'Fresh five-seed author results. Each point is one complete fit; diamond/error bar is mean plus or minus sample seed SD. Separate detail scales; dashed lines are published targets.'}
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[TEMPORAL_RESULTS]]',graph_results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l123/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l123/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0" style="max-width:100%;overflow:auto"><img src="{src}" alt="{caption}"{(' style="min-width:620px;width:100%;max-width:none"' if portable else '')}><figcaption>{caption}</figcaption></figure>')
 fallback='Static trace: at day8, Memo1 → Transfer0 → Person0 gives3 nodes/2 edges. Ignoring availability adds Transfer2:4 nodes/3 edges. At day12 all6 nodes/5 edges are visible.'
 text=text.replace('[[TEMPORAL_WIDGET]]',fallback if portable else '<div id="l123-temporal"></div><noscript><p>'+fallback+'</p></noscript>')
 text=text.replace('[[WARMUP]]','Recall without notes: distinguish entity identity from a graph index, and an event time from a prediction cutoff.' if portable else '<div id="warmup"></div><noscript><p>Recall without notes: what does a primary key identify?</p></noscript>')
 text=text.replace('[[TEACHBACK]]','Explain why every hop uses the root query cutoff, then give a delayed-arrival counterexample.' if portable else '<div id="l123-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','temporal-neighborhood'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0122-reg-construction.html">Lesson 122</a></nav><header><p class="stream-kicker">Year 4 · Quarter 1 · Lesson 123</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','temporal-neighborhood-viz','l123-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0119-year-3-synthesis.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l119','l123')
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
payload={str(seed):{'base64':base64.b64encode((P/f'evidence/l123/paper/seed-{seed}/predictions.npz').read_bytes()).decode(),'sha256':hashlib.sha256((P/f'evidence/l123/paper/seed-{seed}/predictions.npz').read_bytes()).hexdigest()} for seed in range(5)}
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
graph_bytes=(P/'evidence/l123/temporal-graph.json.gz').read_bytes()
full_graph_code="""# PROVIDED harness: use your sampler and audit on the complete real graph.
import gzip,base64,hashlib
raw=base64.b64decode(GRAPH_PAYLOAD)
assert hashlib.sha256(raw).hexdigest()==GRAPH_HASH
real=json.loads(gzip.decompress(raw))
nodes={(kind,i):None if kind not in real['times'] else (real['times'][kind][i],real['times'][kind][i]) for kind,n in real['rows'].items() for i in range(n)}
edges=[]
for key,(source,dest) in real['edges'].items():
    src,rel,dst=key.split('|')
    edges.extend(dict(src=(src,a),dst=(dst,b),kind=rel,stamp=None) for a,b in zip(source,dest))
assert len(nodes)==74063 and len(edges)==338842
for q in real['queries']:
    sample=sample_temporal(nodes,edges,('drivers',q['entity']),q['cutoff'],2)
    audit_sample(nodes,edges,sample)
    assert len(sample['nodes'])==q['nodes'] and len(sample['edges'])==q['edges']
    # Exact set comparison against independently generated oracle identities below.
    key=q['split']+':'+str(q['entity'])+':'+str(q['cutoff'])
    expected=real['query_oracles'][key]
    assert sorted([list(n) for n in sample['nodes']])==expected['nodes']
    assert sample['edges']==expected['edges']
    print(q['split'],q['entity'],q['nodes'],q['edges'],'PASS')
print('Complete graph retained; nine exact neighborhoods. Availability times were not observed.')
""".replace('GRAPH_PAYLOAD',repr(base64.b64encode(graph_bytes).decode())).replace('GRAPH_HASH',repr(hashlib.sha256(graph_bytes).hexdigest()))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 123 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three live functions sample and audit synthetic and real temporal graphs. PROVIDED = inspect; TODO = implement; CHECK = execute unchanged; EXIT = defend. Default execution audits nine queries on the full temporal graph and rescores author predictions; full GPU training is separately gated. Learner status PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((name for name in tasks if 'def '+name+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n'+task_notes[task]+' Implement this function, then run its unchanged CHECK. The toy trace and real F1 temporal harness call your function.' if task else '')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.append(nbf.v4.new_code_cell(checks[tasks[task]]+'\n'+tasks[task]+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · your query-time neighborhoods\n\nPredict the visible nodes at days8,10,12 before execution. The report proves reference execution, not learner mastery.'),nbf.v4.new_code_cell("report=course_run()\nfrom pathlib import Path\nPath('l123-task-report.json').write_text(json.dumps(report,indent=2))\nprint(json.dumps(report,indent=2))"),nbf.v4.new_markdown_cell('## Audit the real author runs\n\nThe embedded predictions are from five fresh full-data fits. They do not replace your own training evidence.'),nbf.v4.new_code_cell(rescore)])
 cells.extend([nbf.v4.new_markdown_cell('## Sample the complete real F1 temporal graph\n\nYour same sampler consumes all released graph rows and timestamp arrays. This input excludes fitted row features. Compare exact nodes and edge identities against the SQL/PyG-verified query oracle. Hashes pin the graph and oracle.'),nbf.v4.new_code_cell(full_graph_code)])
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · every-batch release audit\n\nThis checks identity and root-specific visibility; it cannot observe ingestion histories.'),nbf.v4.new_code_cell((P/'relkit/batch_audit_l123.py').read_text())])
 cells.append(nbf.v4.new_markdown_cell('## Full published implementation and trainer · PROVIDED\n\nUnchanged pinned RelBench release model/trainer from L117, freshly executed in L123. The teaching sampler is exhaustive; the reproduction loader uses bounded uniform fanouts. Source and MIT licenses: [manifest](https://avistian.github.io/relational/labs/sources/l117/manifest.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for name in ['resnet.py','sage_conv.py']:
  cells.append(nbf.v4.new_markdown_cell('### Released primitive · '+name+'\n\nRead-only source of the installed library operator, shown for inspection.\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full five-seed reproduction · OFF by default\n\nRequires the exact pinned GPU runtime in the contract, including native pyg-lib. The notebook gate does not enforce a monetary cap; the supplied Modal runner does reserve bounded compute. No historical environment or live Colab claim is made.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit your graph, implementations, executed artifacts, evidence labels and five written answers from the lesson. Passing checks alone leaves PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l123-{i:03d}'
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
reference="""## Query-time neighborhood contract

| Object | Required check |
|---|---|
| Root entity | Exists and is eligible at query cutoff t |
| Dated row | Event time ≤ t; availability time ≤ t if observed |
| Undated row | Explicit static assumption, not missing-is-safe |
| Relationship | Both endpoints eligible and relationship known by t |
| Reverse relation | Same relationship visibility, reversed message direction |
| Every hop | Preserve root cutoff; do not replace with parent event time |
| Local row | n_id identifies global row; batch identifies query |
| Local edge | Endpoint query IDs agree; e_id preserves original edge |
| Model input | Audit historical feature values and fitted preprocessing separately |

## Worked example

At day8, Memo1(day7) → Transfer0(day4) → Person0 is allowed. Memo0(day12) is future; Transfer2(event5,arrival11) is late. Result3 nodes/2 edges. Day12 query has6 nodes/5 edges. A day7 memo does not need to predate the transfer's day4 when answering a day8 query.

## Evidence boundary

Full F1 graph:74,063 nodes,338,842 edges,72,918 dated rows. Nine exhaustive two-hop neighborhoods match SQL and PyG; five fresh full-data fits additionally audit every sampled batch. No F1 ingestion/mutable-feature history is available. Source preprocessing uses the test-censored database. Node-time checks are not a full pipeline leakage proof.

[Lesson](../lessons/0123-temporal-heterogeneous-graphs.html) · [Protocol](../labs/l123-reproduction.md) · [Temporal audit](../labs/_temporal_audit_l123_results.json).
"""
(R/'reference/temporal-heterogeneous-graphs.html').write_text(document('Temporal heterogeneous graphs · reference',reference))
print('Built L123 lesson, reference and portable notebooks')

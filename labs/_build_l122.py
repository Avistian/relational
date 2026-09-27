"""Build lesson, reference, and standalone notebooks from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0122-reg-construction';TITLE='REG construction: from database keys to a verified graph'
canonical=(P/'relkit/reg_l122.py').read_text();check_source=(P/'_check_l122.py').read_text()
checks={n.name:ast.get_source_segment(check_source,n) for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef)}
task_notes={'key_index':'Goal: map each valid identity to its row. Why: raw key magnitude is not a tensor index. Hint: reject ambiguity before returning a lookup.', 'relation_edges':'Goal: resolve non-null references without changing source-row positions. Why: deleting nulls before numbering shifts edges. Hint: use original source positions and validate parents.', 'construct_reg':'Goal: assemble all typed stores. Why: an edge list alone loses isolated rows and relationship roles. Hint: create node stores first, then one forward/reverse pair per FK column.'}
tasks={'key_index':'check_keys','relation_edges':'check_edges','construct_reg':'check_graph'}
summary=json.loads((P/'evidence/l122/summary.json').read_text())
results='**Fresh author-reference execution: COMPLETE selected released-protocol experiment.**\n\n| Split | Mean MAE | Sample seed SD | Published mean | Verdict |\n|---|---:|---:|---:|---|\n'
for split in ['val','test']:
 m=summary['metrics'][split];results+=f"| {split} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+='\n| Seed | Selected epoch | Validation MAE | Test MAE |\n|---|---:|---:|---:|\n'
for row in summary['seeds']:results+=f"| {row['seed']} | {row['selected_epoch']} | {row['val']:.6f} | {row['test']:.6f} |\n"
results+=f"\nAll **{summary['evaluated_queries']:,}** final predictions were independently scored. Maximum original-model output discrepancy: **{summary['maximum_original_output_error']:.3g}**. Measured pilot plus full-fit worker resource estimate: **USD {summary['worker_resource_usd']:.4f}**, excluding unitemized overhead. The aggregate plan caps reservations plus overhead reserve at USD 10. [Evidence](../labs/evidence/l122/summary.json).\n"
graph_report=json.loads((P/'_graph_audit_l122_results.json').read_text())
graph_results='**Full released F1 topology: PASS.** '+str(graph_report['total_rows'])+' rows; '+str(len(graph_report['forward_relations']))+' forward relation types; '+str(graph_report['directed_edges'])+' directed edges, including reverses.\n\n| Table | Nodes |\n|---|---:|\n'
for name,n in graph_report['rows'].items():graph_results+=f'| {name} | {n:,} |\n'
graph_results+='\nEvery forward endpoint matches an independent SQL join; reverse endpoints match after sorting. [Complete audit](../labs/_graph_audit_l122_results.json).'
captions={'mapping':'Transfer7 maps sender key10 to person row1 and receiver key90 to person row0. Coordinates are local to each node type.', 'routing':'The untrained receiver sum is [7,8,0]. Person300 remains a node despite having no incident edges.', 'paper':'Separate complete RelBench RDL pipeline: row encoders, relative time, typed GraphSAGE and a seed regression head. Reused L117 architecture with fresh L122 execution.'}
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[GRAPH_RESULTS]]',graph_results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l122/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l122/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0" style="max-width:100%;overflow:auto"><img src="{src}" alt="{caption}"{(' style="min-width:620px;width:100%;max-width:none"' if portable else '')}><figcaption>{caption}</figcaption></figure>')
 fallback='Static trace: receiver key90 maps to person row0. Received amounts are [7,8,0]; changing that key to10 gives [2,13,0]. NULL removes only this edge; unknown key999 rejects construction.'
 text=text.replace('[[REG_WIDGET]]',fallback if portable else '<div id="l122-reg"></div><noscript><p>'+fallback+'</p></noscript>')
 text=text.replace('[[WARMUP]]','Recall without notes: distinguish a table from a row, a raw key from an index, and a graph audit from a score reproduction.' if portable else '<div id="warmup"></div><noscript><p>Recall without notes: what does a primary key identify?</p></noscript>')
 text=text.replace('[[TEACHBACK]]','Explain row-permutation invariance without notes. Use the raw key pairs as your evidence.' if portable else '<div id="l122-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0121-history-relational-ml.html">Lesson 121</a></nav><header><p class="stream-kicker">Year 4 · Quarter 1 · Lesson 122</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','reg-construction-viz','l122-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0119-year-3-synthesis.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l119','l122')
payload={str(seed):{'base64':base64.b64encode((P/f'evidence/l122/paper/seed-{seed}/predictions.npz').read_bytes()).decode(),'sha256':hashlib.sha256((P/f'evidence/l122/paper/seed-{seed}/predictions.npz').read_bytes()).hexdigest()} for seed in range(5)}
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
key_bytes=(P/'evidence/l122/key-tables.json.gz').read_bytes()
topology_bytes=(P/'evidence/l122/reg-topology.npz').read_bytes()
full_graph_code="""# PROVIDED harness: learner construct_reg is used on all real key rows.
import gzip,base64,hashlib,io
key_bytes=base64.b64decode(KEY_PAYLOAD)
assert hashlib.sha256(key_bytes).hexdigest()==KEY_HASH
fixture=json.loads(gzip.decompress(key_bytes))
tables={name:pd.DataFrame(v['data'],columns=v['columns'],index=range(v['row_count'])) for name,v in fixture['tables'].items()}
full_graph,unused_feature_names=construct_reg(tables,fixture['schema'])
raw=base64.b64decode(EDGE_PAYLOAD)
assert hashlib.sha256(raw).hexdigest()==EDGE_HASH
oracle=np.load(io.BytesIO(raw),allow_pickle=False)
for name in full_graph.node_types:
    assert full_graph[name].num_nodes==int(oracle['nodes_'+name][0])
for kind in full_graph.edge_types:
    np.testing.assert_array_equal(full_graph[kind].edge_index.numpy(),oracle['|'.join(kind)])
assert sum(full_graph[n].num_nodes for n in full_graph.node_types)==74063
assert sum(full_graph[k].num_edges for k in full_graph.edge_types)==338842
print('PASS: every released F1 row and all26 typed edge stores; key-only construction, not fitted features or temporal safety.')
""".replace('KEY_PAYLOAD',repr(base64.b64encode(key_bytes).decode())).replace('KEY_HASH',repr(hashlib.sha256(key_bytes).hexdigest())).replace('EDGE_PAYLOAD',repr(base64.b64encode(topology_bytes).decode())).replace('EDGE_HASH',repr(hashlib.sha256(topology_bytes).hexdigest()))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 122 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three live functions construct the toy and complete released F1 topology. PROVIDED = inspect; TODO = implement; CHECK = execute unchanged; EXIT = defend. Default execution rebuilds every key row and rescores author predictions; full GPU training is separately gated. Learner status PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((name for name in tasks if 'def '+name+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n'+task_notes[task]+' Implement this function, then run its unchanged CHECK. The toy graph and full released key-table construction call your function.' if task else '')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.append(nbf.v4.new_code_cell(checks[tasks[task]]+'\n'+tasks[task]+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · your row-to-graph construction\n\nPredict receiver endpoints and the received-amount sum before execution. The report proves reference execution, not learner mastery.'),nbf.v4.new_code_cell("report=course_run()\nfrom pathlib import Path\nPath('l122-task-report.json').write_text(json.dumps(report,indent=2))\nprint(json.dumps(report,indent=2))"),nbf.v4.new_markdown_cell('## Audit the real author runs\n\nThe embedded predictions are from five fresh full-data fits. They do not replace your own training evidence.'),nbf.v4.new_code_cell(rescore)])
 cells.extend([nbf.v4.new_markdown_cell('## Rebuild the complete real F1 topology\n\nYour same constructor now consumes all released key rows. This embedded key-only input retains every row; it excludes feature payloads. Compare each edge with the independently SQL-verified topology. Hashes pin the input and oracle.'),nbf.v4.new_code_cell(full_graph_code)])
 cells.append(nbf.v4.new_markdown_cell('## Full published implementation and trainer · PROVIDED\n\nUnchanged pinned RelBench release model/trainer from L117, freshly executed in L122. This differs from the small course model. Source and MIT licenses: [manifest](https://avistian.github.io/relational/labs/sources/l117/manifest.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for name in ['resnet.py','sage_conv.py']:
  cells.append(nbf.v4.new_markdown_cell('### Released primitive · '+name+'\n\nRead-only source of the installed library operator, shown for inspection.\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full five-seed reproduction · OFF by default\n\nRequires the exact pinned GPU runtime in the contract, including native pyg-lib. The notebook gate does not enforce a monetary cap; the supplied Modal runner does reserve bounded compute. No historical environment or live Colab claim is made.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit your graph, implementations, executed artifacts, evidence labels and five written answers from the lesson. Passing checks alone leaves PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l122-{i:03d}'
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
reference="""## REG construction contract

| Input | Graph object | Invariant |
|---|---|---|
| Table | Node type | Distinct local coordinate system |
| Row | Node | Include isolated rows |
| Primary key | Key → row lookup | Unique, non-null; no raw-ID indexing |
| FK column | Typed forward relation | Resolve to parent key; null contributes no edge |
| Reverse store | Reversed endpoints | Preserve relation role, do not invent observations |
| Non-key columns | Row encoder input | Key magnitude excluded by this released model |
| Junction row | Node with multiple relations | Preserve repeated observations and attributes |

## Worked trace

Person keys [90,10,300] map to rows [0,1,2]. Transfer receivers [90,10,90] yield edge_index [[0,1,2],[0,1,0]]. Amounts [5,8,2] sum to [7,8,0]. With person order [300,90,10], destination coordinates become [1,2,1], preserving the same key pairs.

## Audit boundaries

Full F1 topology: 74,063 nodes, 13 forward/13 reverse types, 338,842 directed edges. Compare exact endpoint pairs, not just counts. Local original topology check omits time attributes and uses constant features. Every full GPU fit executes released features and sampling. Construction does not establish temporal validity.

## Reproduction

Five fresh full-data, ten-epoch RelBench v1 Table7 F1 RDL fits. Complete selected release replay; whole-paper and historical identity remain NOT_ESTABLISHED. The default notebook rebuilds full key topology and rescores author predictions; its training gate is OFF.

[Lesson](../lessons/0122-reg-construction.html) · [Protocol](../labs/l122-reproduction.md) · [Graph audit](../labs/_graph_audit_l122_results.json).
"""
(R/'reference/reg-construction.html').write_text(document('REG construction · reference',reference))
print('Built L122 lesson, reference and portable notebooks')

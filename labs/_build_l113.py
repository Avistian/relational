"""Deterministically build the lesson, reference, student and executable solution."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0113-scaling-ogb';TITLE='Scaling OGB: preserve the computation, measure the cost'
canonical=(P/'relkit/scaling_l113.py').read_text()
CAP={'architecture':'Sampling selects an induced graph; GraphSAGE defines the messages. Follow both branches through training and complete-neighbor inference.',
'boundary':'Synthetic scalar trace, weights1 and bias0. Removing node2 changes both the incoming sum and the degree of receiver1.',
'inference':'Every current-layer row is written before the next layer reads it. All incoming neighbors participate, including neighbors outside the receiver chunk.',
'memory':'Sizing calculation for one hidden feature buffer, not measured GPU peak. Three hops, fanout10, before deduplication.',
'results':'Author evidence, separate from learner execution. Only full 50-epoch runs count toward the ten-run target.'}
TASKS={'mean_adjacency':('check_mean','Construct receiver-row sparse incoming means. Preserve repeated-edge multiplicity; isolated receivers have zero neighbor aggregate.'),
'induced_edges':('check_induced','Keep edges with both endpoints selected and remap their global IDs into the given local node order.'),
'selected_epoch':('check_selection','Choose the first maximum validation score among evaluated candidates. Reject empty history.')}
checks_src=(P/'_check_l113.py').read_text();checks={n.name:ast.get_source_segment(checks_src,n) for n in ast.parse(checks_src).body if isinstance(n,ast.FunctionDef)}
def results():
 path=P/'evidence/l113/summary.json'
 if not path.exists():return '**Named ten-run experiment: NOT_RUN.** Data preparation/pilot in progress; no completed-run mean is claimed.'
 r=json.loads(path.read_text());s=f"**Named ten-run experiment: {r['status']}.** {len(r.get('seeds',[]))}/10 completed full schedules.\n\n"
 if r.get('seeds'):
  s+='| Seed | Selected epoch | Validation accuracy | Test accuracy | Training + evaluation seconds |\n|---|---:|---:|---:|---:|\n'
  for a in r['seeds']:s+=f"| {a['seed']} | {a['selected_epoch']} | {a['valid_percent']:.3f}% | {a['test_percent']:.3f}% | {a['seconds']:.1f} |\n"
 s+= '\n'+r.get('interpretation','')+'\n\n[Machine-readable results](../labs/evidence/l113/summary.json).'
 pilot=P/'evidence/l113/pilot/seed-100/result.json'
 if pilot.exists():
  a=json.loads(pilot.read_text());epoch=sum(x['seconds'] for x in a['epochs'])/len(a['epochs']);peak=a['peak_cuda_allocated_bytes']/2**30
  s+=f'\n\n**Measured full-data pilot:** {epoch:.2f}s per training epoch; {a["history"][0]["inference_seconds"]:.2f}s for exact inference; {peak:.2f}GiB peak PyTorch GPU allocation across training and evaluation. Throughput: {196615/epoch:,.0f} supervised nodes/s during training. This two-epoch timing run is not a score reproduction.'

 return s

def bridge_table():
 r=json.loads((P/'_bridge_l113_results.json').read_text());s='**Author-reference teaching run: full arxiv, two epochs, seed113.**\n\n| Training | Steps | Largest batch | Sparse entries per label pass | Time incl. final evaluation | Test accuracy |\n|---|---:|---:|---:|---:|---:|\n'
 for a in r['rows']:s+=f"| {a['regime']} | {a['optimizer_steps']} | {a['max_batch_nodes']:,} | {a['normalized_entries_per_label_pass']:,} | {a['seconds']:.2f} s | {100*a['test']:.2f}% |\n"
 return s+'\nPreprocessing excluded from these times. One CPU thread; recorded workspace runtime. These are short teaching runs, not convergence results.'

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('0089-cluster-gcn.html','0089-sampling-at-scale.html').replace('[[BRIDGE]]',bridge_table()).replace('[[RESULTS]]',results())
 for name in ['MeanSAGE','layerwise_inference']:
  node=next(n for n in ast.parse(canonical).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
  snippet=ast.get_source_segment(canonical,node)
  s=s.replace('[[CODE:'+name+']]', 'See the visible implementation cell below for `'+name+'`.' if portable else '```python\n'+snippet+'\n```')
 for name,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l113/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l113/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure tabindex="0" style="overflow-x:auto"><img style="min-width:640px;max-width:100%" src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 if not portable:s=s.replace(' style="min-width:640px;max-width:100%"','')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="l113-warmup"></div>').replace('[[PREDICT]]','' if portable else '<div id="l113-predict"></div>').replace('[[TEACHBACK]]','' if portable else '<div id="l113-teachback"></div>')
 s=s.replace('[[TRACE]]','**Static control state:** C0 only → output6; C0+C1 → output9. Adding C2 keeps this one-layer output9.' if portable else '<div class="repro-widget" data-scaling-trace></div><noscript><p>C0 only gives6; adding C1 restores the boundary and gives9.</p></noscript>')
 s=s.replace('[[MEMORY]]','**Static sizing state:** 256 roots, fanout10, three hops → 284,416 occurrences and 0.271GiB for one width256 buffer.' if portable else '<div class="repro-widget" data-scaling-memory></div><noscript><p>256 roots and fanout10 across three hops give284,416 node occurrences, or0.271GiB for one width256 float32 buffer.</p></noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"><link rel="stylesheet" href="../assets/reproduction.css"><link rel="stylesheet" href="../assets/scaling.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0112-ogb-gcn-reproduction.html">Lesson 112</a></nav><header><p>Year 3 · Quarter 4 · Lesson 113</p><h1>'+title+'</h1></header>'+render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article>'+(''.join('<script src="../assets/'+name+'.js"></script>' for name in ['scaling-viz','retrieval-pool','retrieval-bank','predict','teachback','l113-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap: install in a fresh environment, then restart if required.
# Core notebook (CPU): torch, numpy, pandas, ogb, scikit-learn.
# Exact cloud versions and compiled sampling wheels: labs/requirements-l113-runtime.txt
# and the commands in labs/l113-reproduction.md. Full reproduction is OFF by default.
import sys, importlib.metadata
print('Python',sys.version)
print({k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','ogb']})
'''
# GCN bridge uses the exact existing model, visibly inlined, not a hidden import.
l112=(P/'relkit/ogb_l112.py').read_text();names=['sha256','normalized_adjacency','GraphConvolution','GCN','load_arxiv','evaluate'];parts={n.name:ast.get_source_segment(l112,n) for n in ast.parse(l112).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names}
# ast source segment omits decorators; evaluate must remain no_grad.
parts['evaluate']='@torch.no_grad()\n'+parts['evaluate']
bridge_src=(P/'_bridge_l113.py').read_text();bridge_fn=next(ast.get_source_segment(bridge_src,n) for n in ast.parse(bridge_src).body if isinstance(n,ast.FunctionDef) and n.name=='bridge')
prepare_src=(P/'_prepare_l113.py').read_text();prepare_inline=prepare_src.split("if __name__=='__main__':")[0];prepare_inline='\n'.join(line for line in prepare_inline.splitlines() if not line.startswith('from relkit.'))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 113 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Three live tasks. Full arxiv bridge (CPU), optional full-products reproduction (GPU). **PENDING_WRITTEN_DEFENSE**.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation · OGB products\n\nThe following code is the canonical model and trainer. METIS and PyG ClusterLoader supply partitioning infrastructure; the graph algebra, loss and training logic remain visible. [MIT source license](https://github.com/snap-stanford/ogb/blob/cf066f93311ab3099cad84d71085d1b0375dcc2e/LICENSE).'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**TODO:** '+TASKS[task][1] if task else '\n\n**PROVIDED.** Trace this code back to the computation in the lesson.')))
  if task and not solution:
   body=next(line for line in body.splitlines() if line.startswith('def '+task+'('))+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   name=TASKS[task][0];cells.append(nbf.v4.new_code_cell(checks[name]+'\n'+name+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('### CHECK · complete computation and inference\n\nThis independent dense oracle tests your live functions. It includes isolated nodes, input gradients, multiple inference chunk sizes and mutated held-out labels.'),nbf.v4.new_code_cell('from types import SimpleNamespace\n'+checks['check_model']+'\ncheck_model(SimpleNamespace(**globals()))\nprint("PASS: complete model and inference")')])
 cells.append(nbf.v4.new_markdown_cell('## L112 bridge · provided GCN, live induced-subgraph task\n\nThe next cells expose L112’s complete model and loader. Only `induced_edges` comes from your live task above. The experiment uses all arxiv nodes, four label-blind random groups and two label passes. Allow roughly 4GiB RAM and a few CPU minutes. It downloads the checksum-pinned 80MiB arxiv archive if needed.'))
 cells.append(nbf.v4.new_code_cell("import urllib.request, zipfile\nimport pandas as pd\nfrom ogb.nodeproppred import Evaluator\nDATA_URL='https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'\nDATA_SHA256='49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276'"))
 for name in names:cells.append(nbf.v4.new_code_cell(parts[name]))
 cells.append(nbf.v4.new_code_cell(bridge_fn))
 cells.append(nbf.v4.new_code_cell("torch.set_num_threads(1)\nx_arxiv,e_arxiv,y_arxiv,s_arxiv,audit_arxiv=load_arxiv(Path('l113-arxiv'))\nfresh=bridge(x_arxiv,e_arxiv,y_arxiv,s_arxiv)\ndisplay(pd.DataFrame(fresh['rows']))\nPath('l113-fresh.json').write_text(json.dumps(fresh,indent=2))\nprint('TEACHING_ONLY: compare computation, not converged accuracy.')"))
 cells.append(nbf.v4.new_markdown_cell('## Full products preparation · PROVIDED, gated OFF\n\nThe 1.48GB archive expands into a large graph. Preparation needs substantial CPU RAM (author cap64GiB); training uses32GiB CPU RAM and a T4. A normal free Colab session may not have enough host RAM. Use the bounded Modal operator for author-equivalent resources. Compiled PyG wheels are necessary only for this optional path. The cell contains the full data/preprocessing path, not a hidden model import.'))
 cells.append(nbf.v4.new_code_cell('RUN_FULL_REPRODUCTION=False\nif RUN_FULL_REPRODUCTION:\n'+''.join('    '+line+'\n' for line in prepare_inline.splitlines())))
 cells.append(nbf.v4.new_code_cell("if RUN_FULL_REPRODUCTION:\n    products_root=Path('l113-products')\n    audit=prepare(products_root)\n    blob=torch.load(products_root/'data.pt',weights_only=False)\n    clusters=torch.load(products_root/'clusters.pt',weights_only=False)\n    products_adj=torch.load(products_root/'adj.pt',weights_only=False)\n    import tempfile\n    output=Path(tempfile.mkdtemp(prefix='l113-paper-'))\n    records=[train_run(blob['data'],clusters,products_adj,blob['split'],output/f'seed-{seed}',seed,50,'cuda',2350) for seed in range(10)]\n    complete=len(records)==10 and all(r['status']=='COMPLETE' for r in records)\n    print('COMPLETE' if complete else 'INCOMPLETE')\n    if complete:\n        report=[]\n        for split_name,target in TARGETS.items():\n            values=np.array([r['selected'][split_name]*100 for r in records])\n            report.append({'split':split_name,'mean_percent':values.mean(),'sample_sd_pp':values.std(ddof=1),'target_percent':target,'verdict':'CLOSE' if abs(values.mean()-target)<=TOLERANCE_PP else 'OUTSIDE_TOLERANCE'})\n        display(pd.DataFrame(report))\n    print('Enforce your external aggregate spend cap; this notebook cannot meter provider billing.')\nelse:\n    print('Full products experiment NOT_RUN in this kernel; see separate author evidence.')"))
 cells.append(nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit the three task functions, `l113-fresh.json`, the compute note, and the four written answers from the lesson. Explain each changed variable before comparing scores. Running the teacher solution is not learner mastery. Ask the teaching agent to grade your defense. **PENDING_WRITTEN_DEFENSE**.'))
 for i,c in enumerate(cells):c.id=f'l113-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for cc,dd in zip(a,b):cc.outputs=dd.outputs;cc.execution_count=dd.execution_count;cc.metadata=dd.metadata
 nbf.write(nb,path)
 if not solution:
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
 elif all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}-solution.html').write_text(html)
ref='''## Sampling and aggregation are separate choices

Cluster sampling chooses an induced batch. OGB products uses SAGE: neighbor mean → learned neighbor transform + separately transformed root. Widths100→256→256→47; ReLU/dropout0.5 after the first two layers. No batch normalization. Training labels only in NLL.

## Induced batch checklist

Choose groups; union their global node IDs; retain every edge with BOTH endpoints selected; remap endpoints to the feature-row order. Preserve edges between selected groups. Degrees are local to the induced graph. Raw self-edges and repeated edges follow released semantics.

## Inference dependency

For one layer, keep previous features fixed. For each receiver chunk, aggregate ALL its incoming neighbors and write global output rows. Finish the entire layer before beginning the next. Dropout off. Chunking receivers is exact; cutting their dependencies is not.

## Frozen published schedule

15000 METIS partitions;32 partitions/batch; Adam0.001;50 epochs; evaluate20,25,...,50; first maximum validation;10 runs. Published valid92.12±0.09%, test78.97±0.33%. Numerical tolerance0.5pp on each mean, only after ten complete runs. Original seeds/partition unknown.

## Compute note

N×H×4 bytes per float32 buffer; products width256 is2.34GiB. Report preparation, partitioning, training and inference separately. GPU allocated-memory measurements exclude CPU storage and some external allocations. Neighbor fanout bound B(1+f1+f1f2+f1f2f3) counts occurrences, before overlap removal. Budget all seeds and retries together.

## Evidence ladder

Dense oracle → released-model check → pilot → completed selected runs → independent checkpoint replay. Each answers a different question. Partial seeds do not establish a ten-run mean. A close score does not establish historical identity.

[Lesson113](../lessons/0113-scaling-ogb.html) · [Protocol and commands](../labs/l113-reproduction.md) · [OGB Table4](https://arxiv.org/html/2005.00687v6#S4.SS1)
'''
(R/'reference/scaling-ogb.html').write_text(document('Scaling OGB · quick reference',ref))
print('Built lesson, reference, student, solution and student HTML')

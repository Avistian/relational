"""Single prose/model source -> lesson, standalone notebooks, reference and solution HTML."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0098-hetero-mini-batching';T='Heterogeneous mini-batching: from typed queries to seed predictions'
result=json.loads((P/'_experiment_l098_results.json').read_text())
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 src='data:image/png;base64,'+base64.b64encode((P/'figures/l098/pipeline.png').read_bytes()).decode() if portable else '../labs/figures/l098/pipeline.svg'
 s=s.replace('[[ARCHITECTURE]]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="Typed database and query cutoff feed two-hop dependency sampling, batch-local tensors, a shared two-layer encoder and seed-only loss, with separate full-neighbor inference and dense-oracle audits."></div><figcaption>Sampling expands backward from seed queries; message passing computes forward toward their predictions.</figcaption></figure>')
 s=s.replace('[[INTERVENTION]]','**Portable intervention:** run the worked calculation after the implementation.' if portable else '<div id="batching-intervention"></div>')
 table='| Training fanout | Mean test accuracy | Mean batch nodes | Mean batch edges | Mean absolute sampled logit gap |\n|---|---:|---:|---:|---:|\n'
 for f in [-1,2,1]:
  rs=[r for r in result['runs'] if r['fanout']==f]
  vals=[sum(r[k] for r in rs)/len(rs) for k in ['test_accuracy','mean_batch_nodes','mean_batch_edges','mean_abs_sampled_logit_gap']]
  table+=f"| {'all' if f==-1 else f} | {vals[0]:.3f} | {vals[1]:.2f} | {vals[2]:.2f} | {vals[3]:.6f} |\n"
 for key,text in [('WARMUP','Recall typed identity and temporal availability before reading.'),('PREDICT','**Predict first:** with fixed weights and all neighbors, should the seed outputs match a full-graph calculation? Explain before reading the result.'),('TEACHBACK','**Teach back:** explain why a seed count differs from a sampled-node count, and why time filtering is a separate boundary.')]:
  s=s.replace('[['+key+']]',text if portable else '<div id="l098-'+key.lower()+'"></div>')
 s=s.replace('[[RESULTS]]',f"**Frozen author results:** 96/96 correctness configurations and 9/9 complete fits. Maximum logit error {max(r['max_logit_gap'] for r in result['audit']):.2e}; maximum gradient error {max(r['max_gradient_gap'] for r in result['audit']):.2e}. Native temporal isolation passed. Your notebook recomputes everything.\n\n"+table)
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
 return s
head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 098 — '+T+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','hetero-graph-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0097-negative-sampling.html">Lesson 97</a> · <a href="../reference/hetero-batching-contract.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 098</p><h1>'+T+'</h1></header>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose())+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in ['hetero-batching-viz','retrieval-pool','retrieval-bank','predict','teachback','l098-lesson'])+'</body></html>')
checks={
'relation_mean':"x=torch.tensor([[2.],[8.],[-4.]],requires_grad=True)\ne=torch.tensor([[0,1,2],[0,0,1]])\ny=relation_mean(x,e,3)\nassert torch.allclose(y,torch.tensor([[5.],[-4.],[0.]]))\ny.sum().backward();assert torch.allclose(x.grad,torch.tensor([[.5],[.5],[1.]]))\nassert relation_mean(x,torch.empty((2,0),dtype=torch.long),0).shape==(0,1)\nprint('PASS: independent means, isolates, empty store, gradients')",
'seed_loss':"b=HeteroData();b['customer'].batch_size=2;b['customer'].y=torch.tensor([0.,1.,1.])\nz=torch.tensor([0.,0.,100.],requires_grad=True)\nloss=seed_loss(z,b);assert abs(loss.item()-0.69314718)<1e-6\nloss.backward();assert torch.allclose(z.grad,torch.tensor([.25,-.25,0.]))\nb['customer'].y[2]=0.;assert abs(seed_loss(z,b).item()-loss.item())<1e-7\nprint('PASS: context labels cannot become extra targets')",
'global_edges':"b=HeteroData();b['orders'].n_id=torch.tensor([42,44,43]);b['customer'].n_id=torch.tensor([7]);b[EDGES[0]].edge_index=torch.tensor([[1,2],[0,0]])\nassert global_edges(b,EDGES[0]).tolist()==[[44,43],[7,7]]\nprint('PASS: local endpoints become typed table row IDs')"}
source=(P/'relkit/batching_l098.py').read_text();tree=ast.parse(source)
bootstrap="""# @colab-bootstrap
# This notebook is self-contained; the native extension must match your torch build.
# Use the exact environment instructions in l098-reproduction.md before this cell.
# No Python fallback is substituted if native sampling is unavailable.
import importlib.metadata as metadata
import torch_geometric.typing as pyg_typing
print({k:metadata.version(k) for k in ['torch','torch-geometric','pyg-lib']})
assert pyg_typing.WITH_PYG_LIB, 'Install the pinned compatible pyg-lib build and restart the kernel.'
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+T+'\n\nTier C synthetic data · native CPU NeighborLoader; all 96 audits and nine fits. No repository imports, hidden trainer or dataset download. PENDING_WRITTEN_DEFENSE.\n\nInstall the environment described in [the reproduction contract](https://avistian.github.io/relational/labs/l098-reproduction.md). Live Colab is NOT_CHECKED; the compiled sampler is required.'),nbf.v4.new_code_cell(bootstrap)]
 cells.extend(nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=\n## )',prose(True)) if x.strip())
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nRead the generated graph and time boundary first. Three functions are your tasks; every other operator and training step is supplied visibly.'))
 prelude='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))
 cells.append(nbf.v4.new_code_cell(prelude))
 for node in tree.body:
  if not isinstance(node,(ast.FunctionDef,ast.ClassDef)):continue
  name=node.name;code=ast.get_source_segment(source,node)
  cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if name in checks else '### PROVIDED · ')+name+'\n\n'+(ast.get_docstring(node) or '')))
  if name in checks and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement the declared contract")'
  cells.append(nbf.v4.new_code_cell(code,metadata={'task':name} if name in checks else {}))
  if name in checks:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+name),nbf.v4.new_code_cell(checks[name])])
 cells.extend([nbf.v4.new_markdown_cell('## Predict → intervene → explain\n\nFor values [2,8,-4], predict the full mean and prefix-two mean. Then inspect actual IDs, component membership and per-type counts from the native loader.'),nbf.v4.new_code_cell("values=torch.tensor([2.,8.,-4.])\nassert values.mean()==2 and values[:2].mean()==5\ng=snapshot(make_graph(98),5)\nb=next(iter(make_loader(g,[7],-1,1)))\nprint({t:b[t].n_id.tolist() for t in TYPES})\nprint('seed count',b['customer'].batch_size)\nprint(temporal_audit())"),nbf.v4.new_markdown_cell('## Full fresh reproduction\n\nRuns all 96 correctness configurations and nine 40-epoch fits. Compare fresh full-neighbor invariants before comparing stochastic optimization traces. Frozen author metrics below are explicitly references, never replacements for fresh computation.'),nbf.v4.new_code_cell("fresh=run_suite()\nfrom pathlib import Path\nPath('l098-fresh.json').write_text(json.dumps(fresh,indent=2)+'\\n')\nassert len(fresh['audit'])==96 and len(fresh['runs'])==9\ntrained=restore_model(fresh['runs'][0])\nwith torch.no_grad(): print('Restored trained model test logits:', trained(snapshot(make_graph(98),5))[20:])\nAUTHOR_TEST_BCE="+repr([r['test_bce'] for r in result['runs']])+"\nprint('BCE gaps from frozen author runs:',[abs(r['test_bce']-a) for r,a in zip(fresh['runs'],AUTHOR_TEST_BCE)])\nprint('Full course suite:',fresh['status'], '; RelBench benchmark:',fresh['relbench_benchmark'])\nprint([(r['seed'],r['fanout'],r['selected_epoch'],r['test_accuracy']) for r in fresh['runs']])"),nbf.v4.new_markdown_cell('## EXIT · PENDING_WRITTEN_DEFENSE\n\nSubmit your three implementations, l098-fresh.json and the five written defenses in section 8. Explain the time boundary, typed ID reconstruction, seed-only loss, weighting of accumulated gradients and remaining RelBench reproduction gaps. Ask the agent to challenge your explanation.')])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 for i,c in enumerate(cells):c.id=f'l098-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if solution:
  html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''# Heterogeneous mini-batching contract

[Lesson 98](../lessons/0098-hetero-mini-batching.html) · [Complete protocol](../labs/l098-reproduction.md)

| Object | Contract |
|---|---|
| Query | Entity type, entity ID, prediction time and query-specific target |
| Input graph | Only features and edges available at prediction time |
| Dependency | Incoming neighbors expanded once per layer |
| Fanout | Per relation, destination and hop; −1 means all |
| Local identity | edge_index endpoints index batch-local type stores |
| Global identity | batch[type].n_id maps local rows to original typed rows |
| Query identity | input_id indexes the input list; entity IDs can repeat |
| Edge identity | e_id indexes the original relation store |
| Loss | First batch[type].batch_size logits and labels only |
| Temporal batch | Separate components for different query histories |
| Full-neighbor parity | Same weights, depth and operators; compare seed outputs |
| Gradient parity | Accumulate B/N-weighted mean losses without optimizer steps |
| Sampling approximation | A one-hop unbiased mean does not imply unbiased final logits |
| Reproduction | Full course toy suite is not full RelBench reproduction |

## Worked trace

orders.n_id=[42,44,43], customer.n_id=[7]: local edge (1,0) restores to (orders 44, customer 7). Three messages 2,8,−4 average to 2; choosing the first two yields 5. Empty neighborhoods yield a zero message; the root transform still acts.

## Before claiming equivalence

Check edge direction, root contributions, hop count, reduction normalization, parameter state, dropout/batch normalization and seed coverage. For time-sensitive queries inspect both directions of every relationship and the availability of preprocessing statistics and features.

Primary sources: [PyG loader](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/neighbor_loader.html), [pinned RelBench example](https://github.com/stanford-star/relbench/blob/584a03d518b2b655580ea8e1cfbbb26bec0a2841/examples/gnn_entity.py). Return tomorrow and reconstruct the loss slice without reading.
'''
(R/'reference/hetero-batching-contract.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Heterogeneous batching contract</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/mpnn-lesson.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built lesson, notebooks, solution HTML and reference')

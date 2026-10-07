"""Build aligned lesson/reference and standalone student/solution notebooks."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0133-hetero-conv-reg';TITLE='Heterogeneous convolution on a relational entity graph'
canonical=(P/'relkit/hetero_l133.py').read_text()
functions={n.name:ast.get_source_segment(canonical,n) for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef)}
checks={n.name:ast.get_source_segment((P/'_check_l133.py').read_text(),n) for n in ast.parse((P/'_check_l133.py').read_text()).body if isinstance(n,ast.FunctionDef)}
tasks={'sum_neighbors':'check_neighbors','relation_output':'check_relation','merge_relations':'check_merge'}
s=json.loads((P/'evidence/l133/summary.json').read_text())
captions={'architecture':'The full driver-position predictor, with the two sums and per-relation root transforms opened inside the graph layer.','arithmetic':'Two distinct relation transforms use the same customer vector; 19 plus negative 6.5 gives 12.5 before normalization.','presence':'An empty relation contributes its root transform and bias. An absent relation is skipped.','scores':'Five fresh ten-epoch full-data fits; mean and sample seed standard deviation versus published means. Split scales differ.'}
results='**Selected released-protocol experiment: COMPLETE.**\n\n| Split | Fresh mean MAE | Sample seed SD | Paper mean | Verdict |\n|---|---:|---:|---:|---|\n'
for k in ['val','test']:
 m=s['metrics'][k];results+=f"| {k} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+='\nAll 6,295 final predictions independently rescored and checked against the original model. Every training/evaluation batch is audited for query isolation, timestamp cutoff and source edges. [Fresh evidence](../labs/evidence/l133/summary.json).\n\nThe first GPU shadow check failed its tight numerical tolerance. The preserved recovery compares cloned layers on CPU float64 using the actual sampled feature values; GPU training is unchanged. This checks arithmetic separately from GPU reduction ordering. [Local behavior checks](../labs/_verify_l133_results.json).'
gate_verified=(P/'_notebook_gpu_l133_results.json').exists() and json.loads((P/'_notebook_gpu_l133_results.json').read_text())['status']=='PASS'
if gate_verified:
 results+='\n\n**Portable full-training path: PASS.** All 24 notebook code cells also executed in an isolated pinned GPU Python namespace, including five fresh ten-epoch fits and five layer audits. Their 6,295 predictions were independently rescored. These validation runs are separate from the primary result table. [Portable gate evidence](../labs/_notebook_gpu_l133_results.json). Live Colab remains NOT_CHECKED.'
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for marker,name in [('NEIGHBOR_CODE','sum_neighbors'),('RELATION_CODE','relation_output'),('MERGE_CODE','merge_relations')]:text=text.replace('[['+marker+']]','```python\n'+functions[name]+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l133/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l133/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 fallback='At customer4: tickets present →12.5; empty →10; absent →19. At customer0: present →8.5; empty →6; absent →7.'
 text=text.replace('[[WIDGET]]',fallback if portable else '<div id="l133-layer" class="rdl-viz"></div><noscript>'+fallback+'</noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','rdl-stack-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0132-identity-aware-message-passing.html">Lesson132</a></nav><header><p class="stream-kicker">Year4 · Quarter2 · Lesson133</p><h1>'+title+'</h1></header>'+body+'</article>'+('<script src="../assets/l133-lesson.js"></script>' if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0130-rdl-checkpoint.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l130','l133')
# The original full trainer remains visible; learner functions audit its first actual layer.
gate=gate.replace('    frozen=dict(epochs=10,seeds=list(range(5)))','''    BaseGateModel=Model
    gate_traces=[]
    class GateModel(BaseGateModel):
        def forward(self,batch,entity_table):
            if self.training and not getattr(self,'_inspected',False):
                self._inspected=True
                def inspect(gnn,args):
                    handle.remove()
                    gate_traces.append(audit_layer(gnn.convs[0],args[0],args[1]))
                handle=self.gnn.register_forward_pre_hook(inspect)
            return super().forward(batch,entity_table)
    Model=GateModel
    frozen=dict(epochs=10,seeds=list(range(5)))''').replace('        NeighborLoader=OriginalLoader','        NeighborLoader=OriginalLoader\n        Model=BaseGateModel').replace('    fresh=reproduction_verdict(packet)',"    assert len(gate_traces)==5 and all(t['status']=='PASS' for t in gate_traces)\n    Path('l133-full/layer-traces.json').write_text(json.dumps(gate_traces,indent=2))\n    fresh=reproduction_verdict(packet)")
payload={}
for seed in range(5):
 root=P/f'evidence/l133/paper/seed-{seed}';raw=(root/'predictions.npz').read_bytes();payload[str(seed)]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads((root/'result.json').read_text())}
replay='''# Recorded author predictions: rescore them; do not call this fresh training.
import io,base64,hashlib,json,statistics,math
from pathlib import Path
import numpy as np
payload = '''+repr(payload)+'''
metrics={'val':[],'test':[]};count=0
for seed,item in payload.items():
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    a=np.load(io.BytesIO(raw));r=item['result']
    assert r['epochs']==10 and len(r['trace'])==10 and all(x['train_queries']==7453 for x in r['trace'])
    for split,n in [('val',499),('test',760)]:
        assert len(set(zip(a[split+'_entity'],a[split+'_time'])))==n
        score=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
        assert abs(score-r['scores'][split])<1e-12
        metrics[split].append(score);count+=n
report=dict(status='PASS',predictions=count,metrics={k:dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for k,v in metrics.items()},tiny_fit=fit_report,full_gate='OFF',learner='PENDING_WRITTEN_DEFENSE')
Path('l133-report.json').write_text(json.dumps(report,indent=2));print(report)
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson133 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three live TODOs, unchanged CHECKs, synthetic optimization and author-evidence replay. Fresh full-data training is gated separately.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell('import copy,json,hashlib,importlib.metadata\nfrom pathlib import Path\nimport torch\ntorch.set_num_threads(1)\nfrom torch_geometric.nn import SAGEConv,HeteroConv')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  # Student explanations retain contracts; implementations belong to the TODO cells.
  if not solution:
   for name in tasks:section=section.replace('```python\n'+functions[name]+'\n```','Implement this operation in the TODO cell below.')
  cells.append(nbf.v4.new_markdown_cell(section))
  for prefix,name in [('## 2','sum_neighbors'),('## 3','relation_output'),('## 4','merge_relations')]:
   if section.startswith(prefix):
    source=functions[name] if solution else functions[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nbf.v4.new_markdown_cell('### TODO · '+name+'\n\nImplement the described contract; keep the CHECK unchanged.'),nbf.v4.new_code_cell(source),nbf.v4.new_code_cell(checks[tasks[name]]+'\n'+tasks[name]+'('+name+')\nprint("PASS: '+name+'")')])
 cells.append(nbf.v4.new_markdown_cell('## Your functions inside a neural computation\n\nAll three TODOs feed the actual layer below. CPU float64 shadow checks compare outputs and gradients. The synthetic fit is separate from the benchmark.'))
 for name in ['explicit_layer','audit_layer','tiny_experiment']:cells.append(nbf.v4.new_code_cell(functions[name]))
 cells.append(nbf.v4.new_code_cell("fit_report=tiny_experiment()\nassert fit_report['status']=='PASS'\nprint(fit_report)"))
 cells.extend([nbf.v4.new_markdown_cell('## Independently rescore author evidence'),nbf.v4.new_code_cell(replay)])
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · complete released model, graph construction and trainer\n\nPinned RelBench commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, MIT license. The following cells contain the load-bearing implementation; no repository imports are required. [Source manifest](https://avistian.github.io/relational/labs/_sources_l133.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for path in ['relkit/batch_audit_l123.py','relkit/checkpoint_l130.py']:cells.append(nbf.v4.new_code_cell((P/path).read_text()))
 for name in ['sage_conv.py','hetero_conv.py','resnet.py','stype_encoder.py']:
  cells.append(nbf.v4.new_markdown_cell('### Read-only pinned primitive · '+name+'\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Optional full five-seed reproduction\n\nOFF by default. Requires the pinned GPU runtime and native pyg-lib. This freshly trains five full fits and uses YOUR layer functions to audit actual first batches. The standalone gate has no monetary guard; the documented Modal runner reserves aggregate cost. The gate and live Colab are NOT_CHECKED for Lesson133 unless separate evidence explicitly says otherwise.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT submission\n\nSubmit your handwritten relation arithmetic, empty/absent explanation, remapped edge indices and reproduction-boundary defense. Author execution leaves learner PENDING_WRITTEN_DEFENSE.')])
 if gate_verified:
  for c in cells:
   if c.cell_type=='markdown':c.source=c.source.replace('The gate and live Colab are NOT_CHECKED for Lesson133 unless separate evidence explicitly says otherwise.','The complete code path passed in a separate pinned GPU Python namespace with five fresh full fits. This default notebook keeps training OFF. Live Colab remains NOT_CHECKED.')
 for i,c in enumerate(cells):c.id=f'l133-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
reference='''## Reconstruct one layer

1. Gather source features using edge row0; scatter-add using edge row1 into all destination rows.
2. Per relation: `lin_l(neighbor_sum) + lin_r(destination)`. Neighbor transform includes bias; root transform is relation-specific.
3. Sum complete relation outputs by destination table.
4. Apply node-wise LayerNorm and ReLU. Repeat for the next layer.

| Input change | Consequence |
|---|---|
| Empty edge tensor | Zero neighbor sum; bias and root term remain |
| Absent relation key | Entire relation skipped |
| Reverse relation | Separate parameters; reverse local endpoints |
| Duplicate edge | Additional message; preserve input multiplicity |
| Table-row permutation | Remap both endpoints; output follows the same permutation |
| Inner mean instead of sum | Changes degree dependence; not the selected experiment |

## Two sums, two kinds of evidence

Neighbor sum reduces incoming edges within a relation. Relation sum reduces outputs entering one destination type. Both sums preserve gradients. Source parity validates the computation; a completed full-data selected experiment validates only its declared protocol. Neither establishes historical identity, all-paper results or learner mastery.

## Audit boundaries

First real minibatch: CPU float64 clones compare layer outputs and convolution/input gradients. Original GPU training remains unchanged. Three semantic mutants, permutation checks and an Adam update are tested locally. Full row-encoder gradient health is outside the layer check.

[Lesson](../lessons/0133-hetero-conv-reg.html) · [Student lab](../labs/0133-hetero-conv-reg.ipynb) · [Reproduction protocol](../labs/l133-reproduction.md) · [PyG HeteroConv](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.HeteroConv.html).
'''
(R/'reference/hetero-conv-reg.html').write_text(document('Heterogeneous convolution · computation reference',reference))
print('Built Lesson133, reference and portable notebooks')

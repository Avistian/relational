"""Build measured lesson, reference and independently executable visible notebooks."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0131-gnn-tabular-stack';TITLE='GNN + tabular encoder stack: trace the forward and backward pass'
canonical=(P/'relkit/stack_l131.py').read_text()
functions={n.name:ast.get_source_segment(canonical,n) for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef)}
check_source=(P/'_check_l131.py').read_text();checks={n.name:ast.get_source_segment(check_source,n) for n in ast.parse(check_source).body if isinstance(n,ast.FunctionDef)}
tasks={'activation_summary':'check_summary','relative_days':'check_days','seed_readout':'check_readout'}
s=json.loads((P/'evidence/l131/summary.json').read_text());t=json.loads((P/'evidence/l131/paper/seed-0/stack-trace.json').read_text())
captions={'architecture':'Measured seed-0 first-batch counts, typed row encoders, time addition, two graph layers and root loss. The backward arrow reaches connected context encoders.', 'time':'Recorded occurrence owners gather their own query cutoffs. Integer seconds become days; no shared first-query clock.', 'activations':'The first three results occurrences and six of 128 channels. Check H + T coordinate by coordinate; later layers mix channels.', 'gradients':'Finite-entry gradient norms for the same batch, weights and random state. Nonfinite entries are counted separately; detach removes row-encoder gradients.', 'scores':'Five fresh ten-epoch full-data fits. Points are seeds; diamonds show mean and sample seed SD; dashed lines are published means. Detail scales differ by split.'}
results='**Fresh selected released-protocol experiment: COMPLETE.**\n\n| Split | Mean MAE | Sample seed SD | Paper mean | Descriptive verdict |\n|---|---:|---:|---:|---|\n'
for split in ['val','test']:
 m=s['metrics'][split];results+=f"| {split} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+='\n| Seed | Selected epoch | Final validation MAE | Test MAE |\n|---|---:|---:|---:|\n'
for row in s['seeds']:results+=f"| {row['seed']} | {row['selected_epoch']} | {row['val']:.6f} | {row['test']:.6f} |\n"
results+=f"\nAll **6,295** final predictions independently scored and compared with the original model. **403,895** sampled query occurrences audited. Successful pilot plus five-fit worker resource estimate **USD {s['worker_resource_usd']:.6f}**; failed pilot and notebook validation are accounted separately within the aggregate budget, with unitemized overhead. [Measured summary](../labs/evidence/l131/summary.json)."
results+='\n\n**Portable full-training gate: PASS.** All 24 notebook code cells executed in a separate pinned GPU environment, including five fresh fits and five stack traces. Its predictions were independently rescored; these validation runs are separate from the primary results above. The preempted attempt is preserved. [Gate verification](../labs/_notebook_gpu_l131_results.json). Live Colab remains NOT_CHECKED.'
trace_table='| Table | Actual row occurrences | Encoded shape |\n|---|---:|---|\n'
for kind,info in t['inputs'].items():trace_table+=f"| {kind} | {info['rows']:,} | [{info['rows']}, 128] |\n"
trace_table+=f"\nThe root head returns **[{t['batch_size']}, 1]**; this first-batch raw L1 loss is **{t['loss']:.6f}**. First three predictions: "+', '.join(f'{v:.4f}' for v in t['predictions'][:3])+'.'
finding=f"**Measured caveat.** The seed-0 original and traced models both contain **{t['gradients']['encoder']['nonfinite']} nonfinite gradient entries**, in `"+'`, `'.join(t['parity']['nonfinite_gradient_parameters'])+f"`. Nonfinite masks match; the maximum finite gradient discrepancy is **{t['parity']['max_gradient_error']:.3g}**. The initial pilot checker stopped on these NaNs. The corrected audit checks mask equality and finite values separately, and retains the original algorithm. This is source parity with an exposed pathology, not a claim of healthy gradients.\n\nThe pinned `StypeEncoder.forward` applies `nan_to_num` **after** feature encoding. A separately executed minimal example, `nan_to_num(w * NaN)`, produces zero forward output but a NaN gradient for w. This explains how finite activations can coexist with nonfinite gradients; the [diagnostic](../labs/_missing_gradient_l131_results.json) is separate from the five-seed benchmark. Repairing the encoder would be a new experimental variant."
def prose(portable=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[TRACE_TABLE]]',trace_table).replace('[[GRADIENT_FINDING]]',finding)
 text=text.replace('[[FORWARD_CODE]]','```python\n'+functions['traced_forward']+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l131/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l131/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 text=text.replace('[[WARMUP]]','Recall the distinction between query roots, context occurrences and table parameters.' if portable else '<div id="warmup"></div><noscript>Recall query roots, context occurrences and table parameters.</noscript>')
 text=text.replace('[[TIME_WIDGET]]','Static arithmetic: correct ages [1,2] days; first-query clock gives [1,-8], which must be rejected.' if portable else '<div class="rdl-viz" id="l131-time"></div><noscript>Static arithmetic: ages [1,2]; wrong owner yields [1,-8], rejected.</noscript>')
 data={k:t[k] for k in ['gradients','detached_gradients','parity']}
 text=text.replace('[[GRAD_WIDGET]]','Static intervention: predictions agree within measured tolerance; detached row encoders have no gradients, while GNN and head retain gradients.' if portable else '<div class="rdl-viz" id="l131-gradient"></div><script type="application/json" id="l131-gradient-data">'+json.dumps(data)+'</script><noscript>Static intervention: detaching encoded rows preserves the forward values and removes encoder gradients.</noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','rdl-stack-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0130-rdl-checkpoint.html">Lesson 130</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Lesson 131</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','stack-trace-viz','l131-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
previous=nbf.read(P/'solutions/0130-rdl-checkpoint.ipynb',as_version=4)
bootstrap=next(c.source for c in previous.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in previous.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l130','l131')
gate=gate.replace('    frozen=dict(epochs=10,seeds=list(range(5)))', '''    OriginalModelClass=Model
    gate_traces=[]
    class InstrumentedModel(OriginalModelClass):
        def forward(self,batch,entity_table):
            if self.training and not getattr(self,'_trace_done',False):
                self._trace_done=True
                gate_traces.append(trace_batch(self,batch,entity_table))
            return super().forward(batch,entity_table)
    Model=InstrumentedModel
    frozen=dict(epochs=10,seeds=list(range(5)))''')
gate=gate.replace('        NeighborLoader=OriginalLoader','        NeighborLoader=OriginalLoader\n        Model=OriginalModelClass')
gate=gate.replace("    fresh=reproduction_verdict(packet)","    assert len(gate_traces)==5 and all(x['status']=='PASS' for x in gate_traces)\n    Path('l131-full/stack-traces.json').write_text(json.dumps(gate_traces))\n    fresh=reproduction_verdict(packet)")
payload={}
for seed in range(5):
 root=P/f'evidence/l131/paper/seed-{seed}';raw=(root/'predictions.npz').read_bytes();payload[str(seed)]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads((root/'result.json').read_text())}
replay='''# PROVIDED: recorded real activations and query clocks, not a fresh neural run.
import io,base64,hashlib,json,statistics,math
from pathlib import Path
import numpy as np
recorded_trace = '''+repr(t)+'''
payload = '''+repr(payload)+'''
# YOUR relative_days checks every recorded timestamp snapshot.
for kind,v in recorded_trace['time_inputs'].items():
    ages=relative_days(torch.tensor(v['cutoff_seconds'],dtype=torch.long),torch.tensor(v['node_seconds'],dtype=torch.long),torch.tensor(v['owner'],dtype=torch.long))
    torch.testing.assert_close(ages,torch.tensor(recorded_trace['stages']['relative_days'][kind]['first_rows']))
# YOUR summary checks actual recorded coordinate samples (not full-tensor means).
for kind in ['results','races']:
    h=torch.tensor(recorded_trace['stages']['row_encoder'][kind]['first_rows'])
    z=torch.tensor(recorded_trace['stages']['time_encoder'][kind]['first_rows'])
    added=torch.tensor(recorded_trace['stages']['time_added'][kind]['first_rows'])
    torch.testing.assert_close(h+z,added)
    print(kind,activation_summary(added))
    print('First two recorded rows, six displayed coordinates:',seed_readout(added,2))
# Rescore all final predictions; this is author evidence replay, not training.
metrics={'val':[],'test':[]};count=0
for seed,item in payload.items():
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    a=np.load(io.BytesIO(raw));r=item['result']
    assert r['epochs']==10 and len(r['trace'])==10 and all(x['train_queries']==7453 for x in r['trace'])
    for split,n in [('val',499),('test',760)]:
        keys=list(zip(a[split+'_entity'],a[split+'_time']));assert len(set(keys))==n
        score=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
        assert abs(score-r['scores'][split])<1e-12;metrics[split].append(score);count+=n
report=dict(status='PASS',predictions=count,metrics={k:dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for k,v in metrics.items()},replay='Author evidence, no fresh training',learner='PENDING_WRITTEN_DEFENSE')
Path('l131-report.json').write_text(json.dumps(report,indent=2));print(report)
'''
fixture='''# CHECK: small complete neural computation, separate from the full F1 evidence.
import pandas as pd
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
torch.set_num_threads(1);torch.manual_seed(131)
data=HeteroData();stats={}
for kind,values in [('drivers',[1.,2.,3.,4.]),('results',[3.,5.,9.,10.])]:
    d=Dataset(pd.DataFrame({'x':values}),col_to_stype={'x':stype.numerical}).materialize()
    data[kind].tf=d.tensor_frame;stats[kind]=d.col_stats
    data[kind].batch=torch.tensor([0,1,0,1]);data[kind].n_id=torch.arange(4)
    data[kind].num_sampled_nodes=[2,1,1]
data['drivers'].seed_time=torch.tensor([864000,1728000]);data['drivers'].y=torch.tensor([3.,7.])
data['results'].time=torch.tensor([777600,1555200,691200,1468800])
data['results','to','drivers'].edge_index=torch.tensor([[0,1,2,3],[0,1,0,1]])
data['drivers','rev_to','results'].edge_index=data['results','to','drivers'].edge_index.flip(0)
for kind in data.edge_types:data[kind].num_sampled_edges=[2,2]
small_model=Model(data,stats,2,128,1,'sum','batch_norm');small_model.train()
before={k:v.clone() for k,v in small_model.state_dict().items()};rng=torch.get_rng_state().clone()
small_trace=trace_batch(small_model,data,'drivers')
assert torch.equal(rng,torch.get_rng_state())
for k,v in before.items():torch.testing.assert_close(v,small_model.state_dict()[k],rtol=0,atol=0)
assert all(p.grad is None for p in small_model.parameters())
assert small_trace['gradients']['encoder']['finite_l2']>0
assert small_trace['detached_gradients']['encoder']['with_gradient']==0
Path('l131-fixture.json').write_text(json.dumps(small_trace,indent=2))
print('PASS: predictions, finite gradients, model state and RNG preserved; detach diagnosed',small_trace['parity'])
# Minimal missing-value autograd mechanism: finite output does not imply finite gradient.
w=torch.tensor(2.,requires_grad=True);safe_output=torch.nan_to_num(w*torch.tensor(float('nan')),nan=0.)
safe_output.backward();assert safe_output.item()==0 and torch.isnan(w.grad)
print('Diagnostic: nan_to_num(w * NaN) = 0, but d(output)/dw is NaN.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 131 · '+TITLE+'\n\n'+('Executed teacher reference' if solution else 'Student lab')+' · PROVIDED: inspect. TODO: implement. CHECK: run unchanged. EXIT: defend. Default execution replays author traces and runs a small full-stack fixture. Fresh full-data training has an explicit gate.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_code_cell('import copy,math,statistics,json,hashlib\nfrom pathlib import Path\nimport torch\n'+checks['rejected'])]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nbf.v4.new_markdown_cell(section))
  number={'## 2':'activation_summary','## 3':'relative_days','## 4':'seed_readout'}
  for prefix,name in number.items():
   if section.startswith(prefix):
    src=functions[name] if solution else functions[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nbf.v4.new_markdown_cell('### TODO · '+name+'\n\nImplement the contract explained above. The CHECK tests errors as well as the happy path.'),nbf.v4.new_code_cell(src),nbf.v4.new_code_cell(checks[tasks[name]]+'\n'+tasks[name]+'('+name+')\nprint("PASS: '+name+'")')])
 cells.extend([nbf.v4.new_markdown_cell('## Replay the measured computation\n\nYour functions inspect recorded real clocks and activation coordinates. This does not regenerate the full neural trace; the visible model fixture and fresh-training gate follow.'),nbf.v4.new_code_cell(replay)])
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · full released model and trainer\n\nPinned RelBench commit 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, MIT license. The table encoders, graph construction, forward pass, loss and checkpoint selection are visible below. [Sources and licenses](https://avistian.github.io/relational/labs/sources/l117/manifest.json).'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 for name in ['traced_forward','gradient_report','trace_batch']:
  cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+name+'\n\nThis calls your live functions. Summaries are detached; prediction tensors retain their graph.'),nbf.v4.new_code_cell(functions[name])])
 cells.extend([nbf.v4.new_markdown_cell('## CHECK · full-stack fixture and state preservation'),nbf.v4.new_code_cell(fixture)])
 for path in ['relkit/batch_audit_l123.py','relkit/checkpoint_l130.py']:
  cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · inherited temporal and experiment audit'),nbf.v4.new_code_cell((P/path).read_text())])
 for name in ['resnet.py','sage_conv.py','hetero_conv.py','stype_encoder.py']:
  cells.append(nbf.v4.new_markdown_cell('### Read-only primitive · '+name+'\n\n```python\n'+(P/'sources/l117/primitives'/name).read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## Full five-seed reproduction gate\n\nOFF by default. Requires the pinned Python3.11/Torch2.5.1 GPU runtime and native pyg-lib. This gate checks versions and hashes, freshly trains all five seeds and records five real forward/backward traces using YOUR functions. The standalone gate has no monetary enforcement; the Modal operator has the aggregate USD10 guard. Live Colab NOT_CHECKED.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSubmit the annotated trace and five written defenses in the lesson. Distinguish recorded activations, your small fixture, new full fits and historical paper parity. PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l131-{i:03d}'
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
reference='''## Forward computation

| Stage | Input → output | Check |
|---|---|---|
| Row encoding | typed columns → N_type × 128 | Separate encoder per table; shared across row occurrences |
| Relative time | cutoff[owner] − node time → days → 128-vector | Correct query owner; no future rows |
| Message passing × 2 | sum neighbors → sum relations → LayerNorm → ReLU | Directed typed edges and root query isolation |
| Root readout | N_drivers × 128 → B × 128 | Loader places roots first |
| Head and loss | B × 128 → B × 1 → mean L1 | One label per query; evaluation clipping separate |

## Gradient diagnosis

Detach a logging copy, not the live row representation. Detaching row vectors can preserve the current output while removing row-encoder gradients. A missing gradient, a zero finite gradient and a nonfinite gradient are distinct. The released first-batch numerical encoder has matched nonfinite gradients: source parity is not proof of healthy optimization. Report finite norms and nonfinite counts separately.

## Instrumentation contract

Compare original and instrumented paths on the same batch, state and RNG; check outputs, losses, finite gradients and nonfinite masks. Use isolated copies and restore RNG so observation does not alter training. Gradients are local sensitivity, not causal attribution.

## Reproduction boundary

Five fresh full-data ten-epoch runs complete the selected RelBench v1 Table7 F1 released-protocol experiment. Whole-paper/historical parity NOT_ESTABLISHED. Original arrival and mutable-feature histories unavailable. Author execution leaves learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0131-gnn-tabular-stack.html) · [Notebook](../labs/0131-gnn-tabular-stack.ipynb) · [Protocol and exact commands](../labs/l131-reproduction.md) · [Primary benchmark paper](https://arxiv.org/html/2407.20060v1).
'''
(R/'reference/gnn-tabular-stack.html').write_text(document('GNN + tabular stack · computation reference',reference))
print('Built Lesson131, reference and portable notebooks')

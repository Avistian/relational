"""Build portable learner package from canonical prose, functions and measured artifacts."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).parent;R=P.parent;S='0134-training-at-scale';TITLE='Training at scale: budget a temporal mini-batch'
canonical=(P/'relkit/scale_l134.py').read_text()
functions={n.name:ast.get_source_segment(canonical,n) for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef)}
checks={n.name:ast.get_source_segment((P/'_check_l134.py').read_text(),n) for n in ast.parse((P/'_check_l134.py').read_text()).body if isinstance(n,ast.FunctionDef)}
tasks={'frontier_bound':'check_bound','audit_queries':'check_audit','profile_summary':'check_summary'}
s=json.loads((P/'evidence/l134/summary.json').read_text());scale=json.loads((P/'evidence/l134/scale/scale.json').read_text())
results='**F1 full selected released-protocol reproduction: COMPLETE.**\n\n| Split | MAE mean ± sample seed SD | Published mean | Verdict |\n|---|---:|---:|---|\n'
for k in ['val','test']:
 m=s['metrics'][k];results+=f"| {k} | {m['mean']:.6f} ± {m['sample_sd']:.6f} | {m['target']:.3f} | {m['verdict']} |\n"
results+=f"\nAll **6,295** final predictions independently rescored; **{s['temporal_audit']['audited_query_occurrences']:,}** query occurrences audited across training and evaluation. Original-model replay maximum difference: {s['maximum_original_output_error']:.3g}. [F1 evidence](../labs/evidence/l134/summary.json).\n\n"
results+=f"**Scale workload: {scale['status']}.** Full graph: **{sum(scale['rows'].values()):,} nodes**, **{scale['directed_edges']:,} directed edges**. Each measured configuration covers the same **2,048 queries** after two warmup update batches.\n\n| Batch / fanouts | Mean sampled nodes | Core queries/s | With audit queries/s | Peak allocated MiB | Peak reserved MiB |\n|---|---:|---:|---:|---:|---:|\n"
for r in scale['configurations']:
 m=r['summary'];results+=f"| {r['batch_size']} / {r['fanouts']} | {sum(x['nodes'] for x in r['batches'])/len(r['batches']):.1f} | {m['queries_per_second']:.1f} | {m['audited_queries_per_second']:.1f} | {m['peak_allocated_bytes']/2**20:.1f} | {m['peak_reserved_bytes']/2**20:.1f} |\n"
results+=f"\nPreprocessing: **{scale['preprocessing_s']:.2f} s**. Peak process host RSS: **{scale['peak_host_rss_kib']/1024:.1f} MiB**, distinct from GPU memory. Measured on **{scale['gpu']}**. Loader construction and warmups are retained in the raw evidence.\n"
results+='\n[Raw scale measurements](../labs/evidence/l134/scale/scale.json). The official database endpoint served bytes differing from the historical registry: the first attempt stopped before training, and the recovery pins the obtained SHA256. This scale lane uses that pinned official archive; historical database identity is NOT_ESTABLISHED. F1 archives match their expected hashes.\n'
if (P/'_notebook_gpu_l134_results.json').exists():
 results+='\n**Portable full-training path: PASS.** All 22 notebook code cells ran in a separate pinned GPU Python namespace, including five fresh complete F1 fits and another 6,295 independently checked predictions. These are validation runs, not replacements for the primary five-seed result. Live Colab frontend remains NOT_CHECKED. [GPU code-path evidence](../labs/_notebook_gpu_l134_results.json).\n'
captions={'batch':'From full host graph to query-owned sampled coordinates, encoded occurrences, typed layers and seed-only loss.','frontier':'A typed no-collision expansion counts 32 occurrences and 30 edges; it is not measured peak memory.','scores':'Five fresh complete F1 runs against published means. Sample seed variability is not a confidence interval.','measurements':'Measured full-topology scale workload with simplified features; sampling, transfer and optimizer step are timed separately.'}
def prose(portable=False,student=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for marker,name in [('BOUND_CODE','frontier_bound'),('AUDIT_CODE','audit_queries'),('SUMMARY_CODE','profile_summary')]:text=text.replace('[['+marker+']]',('Implement this contract in the TODO cell below.' if student else '```python\n'+functions[name]+'\n```'))
 for name,caption in captions.items():
  source='data:image/png;base64,'+base64.b64encode((P/f'figures/l134/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l134/{name}.svg'
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{source}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 fallback='Baseline B=2, fanouts=[3,2]: 32 occurrences, 30 edges, 16 KiB for one width-128 float32 matrix. With B=4 and fanouts=[2,3]: 60 occurrences, 56 edges, 30 KiB.'
 text=text.replace('[[WIDGET]]',fallback if portable else '<div id="l134-budget" class="rdl-viz"></div><noscript>'+fallback+'</noscript>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','rdl-stack-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0133-hetero-conv-reg.html">Lesson133</a></nav><header><p class="stream-kicker">Year4 · Quarter2 · Lesson134</p><h1>'+title+'</h1></header>'+body+'</article>'+('<script src="../assets/l134-lesson.js"></script>' if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
reference='''## A batch has three identities

`n_id`: database row. `batch`: prediction-query owner. `input_id`: task row for the label. Seed supervision uses the first B entity outputs.

## Expansion and memory

For each incoming `(src, relation, dst)`, next-frontier[src] += frontier[dst] × fanout[hop]. Sum roots and all frontiers. Counts are no-collision upper bounds for directional sampling, not distinct global IDs. One float32 hidden tensor costs 4 × N × width bytes; activations, gradients, optimizer and adjacency are additional.

## Temporal validity

Compare each dated node to its OWN seed cutoff. Edges stay within one query owner. Check original global identities too. Undated/static rows and missing ingestion histories are explicit evidence gaps.

## Timing

Synchronize CUDA. Separate loader construction, warmup, sample, transfer, step and audit. Throughput = sum(seed queries)/sum(seconds). Distinguish allocated and reserved GPU memory, host RAM and storage.

## Claim boundaries

Fresh F1 experiment = complete selected released-protocol reproduction. Full-topology rel-stack with simplified features = bounded systems workload. Neither establishes whole-paper parity, historical runtime identity, Colab frontend behavior or learner mastery.

[Lesson](../lessons/0134-training-at-scale.html) · [Exact operators and protocol](../labs/l134-reproduction.md)
'''
(R/'reference/training-at-scale.html').write_text(document('Temporal mini-batch field guide',reference))
prior=nb.read(P/'solutions/0130-rdl-checkpoint.ipynb',as_version=4)
bootstrap=next(c.source for c in prior.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
gate=next(c.source for c in prior.cells if c.cell_type=='code' and 'RUN_FULL_REPRODUCTION = False' in c.source).replace('l130','l134').replace('audit_batch(batch,self.original_graph,self.entity)','inspect_batch(batch,self.original_graph,self.entity)')
payload={}
for seed in range(5):
 root=P/f'evidence/l134/paper/seed-{seed}';raw=(root/'predictions.npz').read_bytes();payload[str(seed)]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads((root/'result.json').read_text())}
replay='''# Author evidence is replayed here; no fresh benchmark training in this default cell.
import io,base64,hashlib,json,math,statistics
from pathlib import Path
payload = '''+repr(payload)+'''
metrics={'val':[],'test':[]};count=0
for seed,item in payload.items():
 raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];a=np.load(io.BytesIO(raw));r=item['result']
 assert r['epochs']==10 and len(r['trace'])==10 and all(x['train_queries']==7453 for x in r['trace'])
 assert r['selected_epoch']==min(r['trace'],key=lambda x:x['val_mae'])['epoch']
 for split,n in [('val',499),('test',760)]:
  assert len(set(zip(a[split+'_entity'],a[split+'_time'])))==n
  score=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
  assert abs(score-r['scores'][split])<1e-12;metrics[split].append(score);count+=n
scale = json.loads('''+repr(json.dumps(scale,separators=(',',':')))+''')
for config in scale['configurations']:
 summary=profile_summary(config['batches'])
 assert summary==config['summary']
 assert [i for r in config['batches'] for i in r['query_indices']]==list(range(2048))
report=dict(status='PASS',predictions=count,metrics={k:dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for k,v in metrics.items()},scale_configurations=len(scale['configurations']),full_gate='OFF',learner='PENDING_WRITTEN_DEFENSE')
Path('l134-report.json').write_text(json.dumps(report,indent=2));print(report)
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson134 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three live functions, real measured evidence, visible full trainer.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell('import math,json,hashlib,importlib.metadata\nfrom pathlib import Path\nimport numpy as np\nimport torch\ntorch.set_num_threads(1)')]
 for section in re.split(r'(?=^## )',prose(True,not solution),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for prefix,name in [('## 3','frontier_bound'),('## 4','audit_queries'),('## 5','profile_summary')]:
   if section.startswith(prefix):
    body=functions[name] if solution else functions[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nb.v4.new_code_cell(body),nb.v4.new_code_cell(checks[tasks[name]]+'\n'+tasks[name]+'('+name+')\nprint("PASS: '+name+'")')])
 cells.extend([nb.v4.new_markdown_cell('## Exercise your functions on a sampled temporal fixture\n\nThe same user has two cutoff-dependent contexts. Your audit must reject the cross-query mutation. Your bound must cover both contexts.'),nb.v4.new_code_cell("schema=[('posts','author','users'),('users','rev','posts')]\nbound=frontier_bound(schema,'users',2,[2,2])\nactual=audit_queries({'users':[1,1],'posts':[4,8]}, {'users':[0,1],'posts':[0,1]},[5,10],[('posts','users',[[0,1],[0,1]])])\nassert actual['nodes']<=bound['node_occurrences']\ntry:\n audit_queries({'users':[1,1],'posts':[4,8]}, {'users':[0,1],'posts':[0,1]},[5,10],[('posts','users',[[1],[0]])])\nexcept ValueError:print('Correctly rejected mixed query')\nelse:raise AssertionError('Query leak')"),nb.v4.new_markdown_cell('## Rescore every author prediction and recompute measured throughput'),nb.v4.new_code_cell(replay)])
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · visible full released F1 implementation\n\nPinned RelBench commit 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639 (MIT). Read these sections to connect the sampled tensors to encoding, graph layers, seed loss and selection. No repository imports are needed.'))
 for chunk in re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nb.v4.new_markdown_cell('### PROVIDED · '+heading),nb.v4.new_code_cell(body.strip())])
 for path in ['relkit/batch_audit_l123.py','relkit/checkpoint_l130.py']:cells.append(nb.v4.new_code_cell((P/path).read_text()))
 cells.append(nb.v4.new_code_cell(functions['inspect_batch'].replace('    from relkit.batch_audit_l123 import audit_batch\n','')))
 cells.extend([nb.v4.new_markdown_cell('## Optional full reproduction\n\nOFF by default. This trains all five full fits in the pinned GPU runtime, using your audit on real batches. This notebook switch has no dollar guard; use the budgeted Modal operator for author runs. Live Colab and this optional notebook training path are NOT_CHECKED unless separate evidence says otherwise.'),nb.v4.new_code_cell(gate)])
 cells.append(nb.v4.new_markdown_cell('## Read the bounded scale operator\n\nThe actual source below makes the projected-column graph build, source checks, warmups, CUDA synchronization and simplified predictor reviewable. It runs through the budgeted operator documented in the reproduction contract. This is a provided systems implementation, not a claim of executed notebook-scale training.\n\n```python\n'+(P/'_run_scale_l134.py').read_text()+'\n```'))
 cells.append(nb.v4.new_markdown_cell('## Submit your EXIT\n\nSend your frontier arithmetic, passing live functions, cutoff defense and measured compute note. Review after 1/7/30 days. Learner status remains PENDING_WRITTEN_DEFENSE.'))
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
 dest=P/('solutions' if solution else '')/f'{S}.ipynb'
 # Deterministic IDs, and retain existing execution only when source is identical.
 if dest.exists():
  old=nb.read(dest,as_version=4)
  if [c.source for c in old.cells]==[c.source for c in notebook.cells]:notebook=old
  elif [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in notebook.cells if c.cell_type=='code']:
   for previous,current in zip([c for c in old.cells if c.cell_type=='code'],[c for c in notebook.cells if c.cell_type=='code']):
    current.outputs=previous.outputs;current.execution_count=previous.execution_count
 if (P/'_notebook_gpu_l134_results.json').exists():
  for c in notebook.cells:
   if c.cell_type=='markdown':c.source=c.source.replace('Live Colab and this optional notebook training path are NOT_CHECKED unless separate evidence says otherwise.','The complete optional code path passed in a separate pinned T4 Python namespace with five fresh full fits. Live Colab frontend remains NOT_CHECKED.')
 for i,c in enumerate(notebook.cells):c.id=f'l134-{i:03d}'
 nb.write(notebook,dest)
print('Built lesson, reference and student/solution notebooks')

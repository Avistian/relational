"""Build lesson, reference and self-contained notebooks from frozen measured artifacts."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).parent;R=P.parent;S='0135-tuning-on-reg';TITLE='Tuning on the REG: spend a budget, defend a decision'
source=(P/'relkit/tuning_l135.py').read_text();functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
checksource=(P/'_check_l135.py').read_text();checks={n.name:ast.get_source_segment(checksource,n) for n in ast.parse(checksource).body if isinstance(n,ast.FunctionDef)}
tasks={'select_configuration':'check_select','reserve_budget':'check_reserve','paired_differences':'check_paired'}
s=json.loads((P/'evidence/l135/summary.json').read_text());frozen=json.loads((P/'evidence/l135/frozen.json').read_text());protocol=json.loads((P/'_protocol_l135.json').read_text())
results='**Author experiment: all 12 search fits and 10 fresh final fits completed.** Winner frozen before final evaluation: `'+s['winner']+'`.\n\n| Condition | Validation MAE mean ± sample SD | Test MAE mean ± sample SD |\n|---|---:|---:|\n'
for c in [s['default'],s['winner']]:
 m=s['metrics'][c];results+=f"| {c} | {m['val']['mean']:.6f} ± {m['val']['sample_sd']:.6f} | {m['test']['mean']:.6f} ± {m['test']['sample_sd']:.6f} |\n"
results+='\n**Published-default comparison:** validation '+s['metrics'][s['default']]['val']['verdict']+'; test '+s['metrics'][s['default']]['test']['verdict']+'. The tuned condition is a course extension.\n'
pair=s['paired_test'];lo,hi=pair['conditional_t95']
results+=f"\nPaired tuned − default test MAE: **{pair['mean_difference']:+.6f}**, sample SD **{pair['sample_sd']:.6f}**; conditional 95% t interval **[{lo:+.6f}, {hi:+.6f}]**.\n\n"
results+=f"Independently rescored **{s['predictions_independently_rescored']:,} predictions** across pilot/search/final artifacts; audited **{s['audited_query_occurrences']:,} query occurrences**. Maximum original-model output difference **{s['maximum_original_output_error']:.3g}**. Primary workers used an estimated **${s['worker_resource_usd']:.3f}**; reserved worker maximum **${s['reserved_worker_upper_usd']:.3f}** plus $3 overhead. Invoice total: **NOT_ITEMIZED**. These are worker measurements, not invoice charges.\n"
if (P/'_notebook_gpu_l135_results.json').exists():
 g=json.loads((P/'_notebook_gpu_l135_results.json').read_text());results+=f"\n**Portable full-training validation: {g['status']}.** The notebook also ran its entire 12-fit search and fresh final comparison in an isolated pinned GPU runtime; its independently selected winner was `{g['packet']['winner']}`. This separate rerun gave a paired mean difference of {g['packet']['paired']['mean_difference']:+.4f} MAE; individual scores were not identical. Fixed seeds do not guarantee bitwise deterministic GPU training. These validation runs are not pooled into the primary comparison. This additional check used an estimated ${g['resource_usd']:.3f}; combined measured worker resources are ${s['worker_resource_usd']+g['resource_usd']:.3f}. All worker reservations plus overhead total $7.876, below the $10 cap. [Execution evidence](../labs/_notebook_gpu_l135_results.json). Live Colab remains NOT_CHECKED.\n"
if pair['mean_difference']<0:
 interpretation=f"The frozen winner reduced mean test MAE by {-pair['mean_difference']:.4f} on this task. "
else:interpretation=f"The frozen winner increased mean test MAE by {pair['mean_difference']:.4f} on this task, despite winning the search validation comparison. "
interpretation+=('The conditional interval crosses zero, so these five fits do not resolve the direction reliably. ' if lo<=0<=hi else 'The conditional interval excludes zero; its scope is still this fixed task and selected configuration. ')
interpretation+='The experiment is complete even if tuning fails to help. Keep the observed result; do not retune after seeing test. Search seed spread and later validation replay are shown separately from the final test pairs.'
captions={'space':'Frozen grid: six configurations, two search seeds and ten epochs per fit. Full data and model width stay fixed.','selection':'Illustrative nested selection: first pick a validation checkpoint per fit, then average both seeds per configuration.','budget':'Worst-case primary worker reservations and overhead; extra notebook checks consume remaining headroom.','search':'Measured selected validation MAE for both search seeds; diamonds show the means used to freeze the winner.','paired':'Measured final seed pairs and signed test differences; the interval covers fit-seed variation only.'}
def prose(portable=False,student=False):
 text=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[INTERPRETATION]]',interpretation)
 for marker,name in [('SELECT_CODE','select_configuration'),('BUDGET_CODE','reserve_budget'),('PAIRED_CODE','paired_differences')]:text=text.replace('[['+marker+']]', 'Implement the contract in the TODO cell below; use its CHECK before continuing.' if student else '```python\n'+functions[name]+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l135/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l135/{name}.svg'
  figure_style=' style="margin:1rem 0;overflow-x:auto"' if portable else ''
  image_style=' style="min-width:580px;max-width:100%;height:auto"' if portable else ''
  text=text.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"{figure_style}><img src="{src}" alt="{caption}"{image_style}><figcaption>{caption}</figcaption></figure>')
 for marker,id,baseline in [('SELECT_WIDGET','l135-selection','Illustrative baseline: A mean 3.0, B mean 2.5 → B wins. Change A’s second seed from 4 to 1 → A mean 1.5, so A wins.'),('BUDGET_WIDGET','l135-budget','Illustrative existing commitments $3 + overhead $3: 12 new workers reserve $2.437776 → allow; 20 reserve $4.062960 → refuse.')]:
  text=text.replace('[['+marker+']]',baseline if portable else '<div class="tuning-control" id="'+id+'"></div><noscript>'+baseline+'</noscript>')
 text=text.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>')
 text=text.replace('[[TEACHBACK]]','Explain in your own words why a completed tuning experiment can still fail to establish general superiority.' if portable else '<div id="l135-teachback"></div>')
 if portable:
  text=text.replace('](../','](https://avistian.github.io/relational/');text=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
 return text

def document(title,text,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(text)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','tuning-budget-viz'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','tuning-budget'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0134-training-at-scale.html">Lesson 134</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Lesson 135</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
reference='''## Freeze before fitting

Record data/source/runtime, split, search space, seed lists, per-trial schedule, metric, tie-break and budget. Equal trials or epochs do not imply equal time, dollars or FLOPs.

## Two selection steps

Within each fit: first lowest validation checkpoint. Across configurations: mean of those selected scores over the exact planned seed set. Reject missing, duplicate or unplanned trials. Search has no test path. Freeze the decision and artifact hashes before final dispatch.

## Budget equation

Committed + workers × timeout × resource rate + overhead ≤ cap. Reserve before launch. Failed trials cost money too. Keep resource estimates, maximum reservations and invoice charges separate.

## Final comparison

Fresh paired seeds, validation-selected checkpoints, frozen test evaluation. Δs = tuned MAE − default MAE; negative favors tuning. Match by seed before subtracting. Report every pair, mean and sample SD. A seed interval excludes search, split and dataset uncertainty.

## Evidence boundary

Complete selected baseline reproduction ≠ whole-paper reproduction. Complete grid ≠ exhaustive hyperparameter universe. Fresh seeds ≠ fresh test split. A disappointing test result is a valid experimental outcome.

[Lesson](../lessons/0135-tuning-on-reg.html) · [Commands and protocol](../labs/l135-reproduction.md) · [RelBench Appendix B.2](https://arxiv.org/html/2407.20060v1#A2.SS2)
'''
(R/'reference/tuning-on-reg.html').write_text(document('REG tuning field guide',reference))
prior=nb.read(P/'solutions/0133-hetero-conv-reg.ipynb',as_version=4)
bootstrap=next(c.source for c in prior.cells if c.cell_type=='code' and '@colab-bootstrap' in c.source)
payload={}
for path in sorted((P/'evidence/l135').glob('**/predictions.npz')):
 raw=path.read_bytes();payload[str(path.parent.relative_to(P/'evidence/l135'))]=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),result=json.loads((path.parent/'result.json').read_text()))
replay='''# PROVIDED: immutable author evidence; not a new training run.
import io,base64,hashlib,json,math,statistics
from pathlib import Path
AUTHOR_PAYLOAD = '''+repr(payload)+'''
search=[];final={};count=0
for key,item in AUTHOR_PAYLOAD.items():
    raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    a=np.load(io.BytesIO(raw));r=item['result'];phase,config,_=key.split('/')
    assert r['selected_epoch']==min(r['trace'],key=lambda t:t['val_mae'])['epoch']
    for split,score in r['scores'].items():
        n=len(a[split+'_pred']);assert len(set(zip(a[split+'_entity'],a[split+'_time'])))==n
        actual=math.fsum(abs(float(x)-float(y)) for x,y in zip(a[split+'_pred'],a[split+'_target']))/n
        assert abs(actual-score)<1e-12;count+=n
    if phase=='search':
        assert set(r['scores'])=={'val'}
        search.append(dict(config=config,seed=r['seed'],selection_mae=r['selection_mae']))
    if phase=='final':final.setdefault(config,[]).append(dict(seed=r['seed'],mae=r['scores']['test']))
decision=select_configuration(search,[c['id'] for c in PROTOCOL['configurations']],PROTOCOL['search_seeds'])
pairs=paired_differences(final[PROTOCOL['default']],final[decision['winner']])
assert decision['winner']=='''+repr(s['winner'])+'''
assert abs(pairs['mean_difference']-'''+repr(pair['mean_difference'])+''')<1e-12
committed=reserve_budget([],23,900,PROTOCOL['rate_usd_second'],3,10)
print('All author predictions rescored:',count)
print('Frozen winner:',decision['winner'],'; paired result:',pairs)
print('Primary worker maximum reservation:',committed)
Path('l135-report.json').write_text(json.dumps(dict(status='PASS',predictions=count,winner=decision['winner'],paired=pairs,learner='PENDING_WRITTEN_DEFENSE'),indent=2))
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 135 · '+TITLE+'\n\n'+('Executed teacher reference' if solution else 'Student lab')+' · Tier B: full released relational data. Three live experiment-control tasks. Default cells rescore author evidence; full training is separately gated.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell('import math,statistics,json,hashlib\nfrom pathlib import Path\nimport numpy as np\nimport torch\ntorch.set_num_threads(1)\nPROTOCOL = '+repr(protocol))]
 for section in re.split(r'(?=^## )',prose(True,not solution),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for prefix,name in [('## 3','select_configuration'),('## 4','reserve_budget'),('## 5','paired_differences')]:
   if section.startswith(prefix):
    body=functions[name] if solution else functions[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nb.v4.new_code_cell(body),nb.v4.new_code_cell(checks[tasks[name]]+'\n'+tasks[name]+'('+name+')\nprint("PASS: '+name+'")')])
 cells.extend([nb.v4.new_markdown_cell('## CHECK · Use all three functions on complete author evidence\n\nYou are independently replaying recorded predictions, not training a new model or claiming mastery. Reversing rows does not change keyed pairing.'),nb.v4.new_code_cell(replay)])
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · Full visible RDL implementation\n\nThe following annotated sections preserve the pinned released basic model (RelBench commit 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, MIT). Typed feature encoding precedes relative-time addition, two relation-specific GraphSAGE layers and a seed-only head. Our selected lane uses sum neighbor and relation aggregation, width 128, two layers, mean L1 loss and Adam. The later trainer exposes the learning rate and fanouts while leaving those mechanisms fixed.'))
 chunks=re.split(r'^# %% ',(P/'relkit/rdl_l117.py').read_text(),flags=re.M)[1:]
 for chunk in chunks:
  heading,body=chunk.split('\n',1)
  if heading.startswith('Full released-protocol training loop'):continue
  cells.extend([nb.v4.new_markdown_cell('### PROVIDED · '+heading),nb.v4.new_code_cell(body.strip())])
 cells.append(nb.v4.new_code_cell((P/'relkit/batch_audit_l123.py').read_text()))
 train=(P/'relkit/tuning_train_l135.py').read_text().replace('from relkit.rdl_l117 import Model, get_node_train_table_input, NeighborLoader, seed_everything, CONFIG as BASE_CONFIG','from torch_geometric.loader import NeighborLoader\nfrom torch_geometric.seed import seed_everything\nBASE_CONFIG = '+repr(dict(channels=128,num_layers=2,batch_size=512,lr=.005,epochs=10,fanout=[128,64],aggr='sum',temporal_strategy='uniform'))).replace('    from relkit.batch_audit_l123 import audit_batch\n','')
 cells.extend([nb.v4.new_markdown_cell('### PROVIDED · Full trainer and information boundary\n\nTrace where `evaluate_test=False` restricts tables and loaders. `configuration` controls the actual optimizer learning rate and neighbor sampler. Each batch is audited. First-best validation state is saved, restored and compared with the original model on identical sampled batches.'),nb.v4.new_code_cell(train)])
 constants='TEXT_SPEC = '+repr(json.loads((P/'sources/l117/text_model.json').read_text()))+'\nORIGINAL_MODEL_SOURCE = '+repr((P/'sources/l117/model.py').read_text())
 cells.extend([nb.v4.new_markdown_cell('## Optional full search and reproduction\n\nOFF by default. This runs all twelve search fits and then both final conditions, using your selection and pairing functions and the visible model/trainer. It needs the exact compatible GPU runtime in the reproduction contract. The Modal operator adds enforceable dollar reservations; this notebook switch does not enforce a dollar cap. Search is complete before test access is enabled. Original reference source is embedded as a parity oracle.'),nb.v4.new_code_cell(constants),nb.v4.new_code_cell((P/'_notebook_gate_l135.py').read_text())])
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
 dest=P/('solutions' if solution else '')/f'{S}.ipynb'
 if dest.exists():
  old=nb.read(dest,as_version=4)
  # Retain outputs when code is unchanged; prose-only refresh does not invent execution.
  oldcode=[c for c in old.cells if c.cell_type=='code'];newcode=[c for c in notebook.cells if c.cell_type=='code']
  if [c.source for c in oldcode]==[c.source for c in newcode]:
   for a,b in zip(newcode,oldcode):a.outputs=b.outputs;a.execution_count=b.execution_count
 for i,c in enumerate(notebook.cells):c.id=f'l135-{i:03d}'
 nb.write(notebook,dest)
print('Built lesson, reference, student and solution notebooks')

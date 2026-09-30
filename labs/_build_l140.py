"""Canonical L140 lesson, reference and portable live-function notebooks."""
import ast,base64,hashlib,io,json,re,zlib
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l140';S='0140-rdl-reproduction-checkpoint';TITLE='Defend a two-task RDL reproduction'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
funcs=defs(P/'relkit/checkpoint_l140.py');checks=defs(P/'_check_l140.py');rank=defs(P/'relkit/amazon_l138.py')['rank_auc']
summary=json.loads((E/'training.json').read_text())
results='| Task / split | Fresh AUROC % ± sample SD | Paper AUROC % | Score | Protocol |\n|---|---:|---:|---|---|\n'
for name,t in summary['tasks'].items():
 for split,v in t['metrics'].items():results+=f"| {name} / {split} | {100*v['mean']:.4f} ± {100*v['sample_sd']:.4f} | {100*v['paper_target']:.2f} | {v['score']} | {v['protocol']} |\n"
count=sum(t['verified_predictions'] for t in summary['tasks'].values())
results+=f'\n**Ten fresh full released-protocol fits; {count:,} held-out predictions independently rescored.** Amazon protocol remains gapped. Historical identity and whole-paper reproduction: **NOT_ESTABLISHED**.\n'
captions={'architecture':'Two concrete relational paths share the encoder → temporal GNN → head computation. The scalar neighbor example is illustrative. Task-specific fanout, aggregation and optimizer settings remain distinct.','selection':'Fresh measured validation curves for every seed. Dots mark first maxima; no test curve is used for selection.','evidence':'Three evidence gates. The 0.2pp gap is illustrative; the Amazon training-count discrepancy is real.','scores':'Fresh L140 seed points, means and sample SD. Dashed lines are paper means. Separate vertical scales expose variance; the Amazon protocol gap remains.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l140/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l140/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on a narrow screen.</figcaption></figure>')
 for marker,id,fallback in [('SELECTION_WIDGET','l140-selection','Synthetic baseline: epoch2 wins; epoch3 ties validation and must not replace it.'),('GATES_WIDGET','l140-gates','Hypothetical baseline:5/5 seeds,0.2pp gap,archive mismatch → COMPLETE / CLOSE / GAPPED; historical identity NOT_ESTABLISHED.')]:s=s.replace('[['+marker+']]',fallback if portable else f'<div class="tuning-control" id="{id}"></div><noscript>{fallback}</noscript>')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write the defense before asking the agent to assess your checkpoint.' if portable else '<div id="l140-teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','reproduction-gates-viz'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 140 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/tuning-budget.css"><link rel="stylesheet" href="../assets/reproduction-gates.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0139-healthcare-trial.html">Lesson 139</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Checkpoint 140</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Reproduction contract
Freeze source, archives, query population, objective, fanouts, optimizer, schedule, seeds, metric and first-best validation selection. A released epoch cap may be shorter than one pass through all training rows. No test-based tuning or discarded seeds.

## Three functions
`select_checkpoint`: first maximum finite validation AUROC over complete ordered epochs. `align_predictions`: identical unique (entity,cutoff) sets, finite probabilities, canonical query order. `reproduction_verdict`: complete seed set before mean/sample SD and closeness; protocol and historical identity are separate.

## Evidence boundaries
±1pp is descriptive, not equivalence. SD over five seeds conditions on fixed data/preprocessing. A temporal sampler audit checks released timestamps, not unknown arrival/version history. Amazon has24,172 fewer training queries than the paper; clinical timestamps are inferred. Reusing graph preprocessing is distinct from reusing trained weights. Default notebook re-scoring is distinct from fresh training. Author completion is distinct from learner defense.

## Fresh measured results
'''+results+'''\n[Lesson](../lessons/0140-rdl-reproduction-checkpoint.html) · [Protocol and commands](../labs/l140-reproduction.md) · [Paper Table6](https://arxiv.org/html/2407.20060v1#A2.SS1).
'''
(R/'reference/rdl-reproduction-checkpoint.html').write_text(doc('Two-task reproduction — reference',reference))
# One compact packet: shared query keys/targets once, probabilities once per seed.
arrays={};histories={};targets={}
for task,entity,source in [('amazon','customer',P/'evidence/l138/recency_predictions.npz'),('trial','study',P/'evidence/l139/queries.npz')]:
 base=np.load(source);histories[task]={};targets[task]={}
 for split in ['val','test']:
  for name,old in [('entity',entity),('time','time'),('target','target')]:arrays[f'{task}_{split}_{name}']=base[f'{split}_{old}']
  targets[task][split]=summary['tasks'][task]['metrics'][split]['paper_target']
 for seed in range(5):
  r=json.loads((E/f'{task}/seed-{seed}/result.json').read_text());histories[task][str(seed)]=r['history']
  z=np.load(E/f'{task}/seed-{seed}/predictions.npz')
  for split in ['val','test']:
   for name in ['entity','time','target']:assert np.array_equal(z[f'{split}_{name}'],arrays[f'{task}_{split}_{name}'])
   arrays[f'{task}_{split}_s{seed}']=z[f'{split}_pred']
 buftrace=np.load(E/f'{task}/seed-0/sampled_trace.npz')
 for name in buftrace.files:arrays[f'{task}_trace_{name}']=buftrace[name]
buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue();compressed=zlib.compress(json.dumps(histories,separators=(',',':')).encode())
packet='''# PROVIDED: immutable fresh author evidence, not student training output.
PACKET=base64.b64decode('''+repr(base64.b64encode(raw).decode())+''')
assert hashlib.sha256(PACKET).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
DATA=np.load(io.BytesIO(PACKET))
HISTORIES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(compressed).decode())+''')))
TARGETS='''+repr(targets)+'\n'
bootstrap='''# @colab-bootstrap — this notebook contains all lesson code and evidence.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
# Colab's default Torch can execute the small fixture. Full reproduction requires
# the exact pinned runtime documented in l140-reproduction.md, plus host RAM.
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','sentence-transformers==3.3.1'])
from pathlib import Path
import base64,hashlib,io,json,zlib
import numpy as np
import pandas as pd
from IPython.display import display
RUN_FULL_REPRODUCTION=False
FULL_TASKS=['amazon','trial'];FULL_SEEDS=list(range(5))
PREPARED_ROOTS={}  # Empty means full fresh materialization; reuse only verified directories.
'''
selection_run='''selected=[]
for task,seeds in HISTORIES.items():
    for seed,history in seeds.items():
        epoch=select_checkpoint(history)
        assert epoch==1+int(np.argmax([r['val']['roc_auc'] for r in history]))
        selected.append(dict(task=task,seed=int(seed),epoch=epoch,selection_auc=history[epoch-1]['val']['roc_auc']))
display(pd.DataFrame(selected))
'''
alignment_run='''rows=[];rng=np.random.default_rng(140)
from sklearn.metrics import roc_auc_score
for task in ['amazon','trial']:
    for split in ['val','test']:
        prefix=f'{task}_{split}_'
        keys=list(zip(DATA[prefix+'entity'],DATA[prefix+'time']))
        target=DATA[prefix+'target'];order=rng.permutation(len(keys))
        for seed in range(5):
            pred=align_predictions(keys,[keys[i] for i in order],DATA[prefix+f's{seed}'][order])
            score=rank_auc(target,pred)
            assert abs(score-roc_auc_score(target,pred))<1e-12
            rows.append(dict(task=task,split=split,seed=seed,auc=score,queries=len(keys)))
display(pd.DataFrame(rows));print('All real prediction vectors shuffled, aligned and independently scored')
'''
verdict_run='''verdicts=[]
for task in ['amazon','trial']:
    for split in ['val','test']:
        values={r['seed']:r['auc'] for r in rows if r['task']==task and r['split']==split}
        v=reproduction_verdict(values,list(range(5)),TARGETS[task][split],.01,task!='amazon')
        verdicts.append(dict(task=task,split=split,**v))
display(pd.DataFrame(verdicts))
assert all(v['execution']=='COMPLETE' for v in verdicts)
print('Scores are descriptive. Historical identity remains NOT_ESTABLISHED.')
# Real sampled timestamps retain the owning query cutoff.
node_occurrences=0
for task in ['amazon','trial']:
    for key in DATA.files:
        if key.startswith(task+'_trace_') and key.endswith('_times'):
            owners=DATA[key[:-6]+'_owners'];times=DATA[key]
            assert (times<=DATA[task+'_trace_cutoffs'][owners]).all()
            node_occurrences+=len(times)
print('Independently checked sampled timestamped node occurrences:',node_occurrences)
'''
model=(P/'relkit/rdl_l117.py').read_text().split('# %% Full released model dependencies (PROVIDED)',1)[1].split('# %% Full released-protocol training loop (PROVIDED)',1)[0]
model_parts=re.split(r'(?=^# %% )',model,flags=re.M)
full=(P/'_full_l140.py').read_text().split("if __name__=='__main__':",1)[0].replace('from relkit.rdl_l117 import Model,make_pkey_fkey_graph,get_node_train_table_input\n','')
# Split full trainer at named functions without altering any AST.
full_parts=re.split(r'(?=^def (?:sha|materialize|full_run)\()',full,flags=re.M)
fixture=(P/'_fixture_l139.py').read_text().split("if __name__=='__main__':",1)[0]
original=(P/'sources/l140/examples__model.py').read_text()
gate='''# Full fresh training is separate from the default artifact audit.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    required={'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}
    for name,version in required.items():assert md.version(name).split('+')[0]==version,(name,'requires pinned runtime')
    assert torch.cuda.is_available(),'GPU and sufficient host RAM required'
    original=Path('sources/l140/examples__model.py');original.parent.mkdir(parents=True,exist_ok=True);original.write_text(ORIGINAL_MODEL)
    for task in FULL_TASKS:
        prepared=Path(PREPARED_ROOTS[task]) if task in PREPARED_ROOTS else materialize(Path('l140-prepared')/task,task)
        for seed in FULL_SEEDS:full_run(Path('l140-full')/task/f'seed-{seed}',task,seed,prepared)
else:print('Fresh full training NOT_RUN in default notebook; author predictions independently rescored above')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson140 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED: inspect/run; TODO: implement; CHECK: run unchanged; EXIT: defend. Tier B: complete released relational task evidence. Default CPU execution audits artifacts; the post-EXIT GPU gate trains fresh models.'),nb.v4.new_markdown_cell('## Concept recap\n\nA query is an entity at a cutoff; a prediction must retain both coordinates. Validation chooses weights; test evaluates the frozen choice. A reproduction report asks separately whether all planned runs finished, whether their metric is close, and whether the available protocol evidence matches.\n\nWorked example: validation [.70,.75,.75] chooses epoch2 even when epoch3 has a better test score. A report with only seeds0–3 cannot summarize a planned five-seed result. With all five present, a close score still cannot repair a changed training population.\n\nThe default evidence was generated by the author in fresh L140 runs. Your live functions select, align and judge those artifacts; default execution is not fresh training or proof of mastery.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell('# PROVIDED: tie-aware rank arithmetic\n'+rank),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload'],'jupyter':{'source_hidden':True}})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  cells.append(nb.v4.new_markdown_cell(section));task=None
  if section.startswith('## 3'):task=('select_checkpoint','check_selection',selection_run)
  if section.startswith('## 4'):task=('align_predictions','check_alignment',alignment_run)
  if section.startswith('## 5'):task=('reproduction_verdict','check_verdict',verdict_run)
  if task:
   name,check,run=task
   cells.append(nb.v4.new_code_cell('# '+('SOLUTION' if solution else 'TODO')+'\n'+(funcs[name] if solution else funcs[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")')))
   cells.append(nb.v4.new_code_cell('# CHECK\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")'))
   cells.append(nb.v4.new_code_cell(run))
 cells.append(nb.v4.new_markdown_cell('## PROVIDED · full model and graph construction\n\nThese are the executed released model primitives, with the unused recommendation-only readout omitted. Read the typed encoders, temporal ownership, relation convolution, root selection and head. MIT source attribution follows. The next cell group reconstructs graph edges and task transforms.'))
 license='\n'.join('# '+x for x in (P/'sources/l140/LICENSE').read_text().splitlines())
 for i,part in enumerate(model_parts):
  cells.append(nb.v4.new_markdown_cell(['### Column, row and temporal encoders; relation message passing','### Root predictor and task head','### Full database graph and query transforms'][i]))
  cells.append(nb.v4.new_code_cell((license+'\n' if i==0 else '')+part))
 cells.append(nb.v4.new_code_cell(fixture+'\nFIXTURE,_,_,_=neural_fixture(Model)\nSUM_FIXTURE,_,_,_=neural_fixture(lambda *args: Model(*args[:5],aggr="sum",norm="batch_norm"))\nprint(FIXTURE);print(SUM_FIXTURE)'))
 cells.append(nb.v4.new_markdown_cell('## NEXT STEP · full released-protocol reproduction\n\nThe code below includes full archive preparation and training; no small-data substitute is hidden. Both task recipes are fixed in `TASKS`. Fresh Amazon preparation needs roughly128GiB host memory; cached training workers use64GiB. Five seeds on both tasks can take substantial time. Use the budgeted Modal operator for author execution; the notebook gate does not enforce billing. [Exact runtime and commands](https://avistian.github.io/relational/labs/l140-reproduction.md).\n\n`materialize` downloads checksum-pinned archives and computes full graph/text features. `full_run` initializes fresh weights, checks query ownership, selects only on validation and compares a saved checkpoint with the original source model.'))
 for part in full_parts:cells.append(nb.v4.new_code_cell(part))
 cells.append(nb.v4.new_code_cell('ORIGINAL_MODEL='+repr(original),metadata={'tags':['data-payload']}))
 cells.append(nb.v4.new_code_cell(gate))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',selected_histories=len(selected),rescored_predictions=sum(r['queries'] for r in rows),verdicts=verdicts,neural_fixture=FIXTURE['status'],fresh_training='EXECUTED' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l140-report.json').write_text(json.dumps(report,indent=2));display(pd.DataFrame(verdicts));print('Author evidence checked; submit your written defense.')"))
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l140-{i:03d}'
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);a=[c for c in cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   notebook.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nb.write(notebook,path)
print('Built L140 lesson/reference/student/solution')

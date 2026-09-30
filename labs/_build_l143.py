"""Canonical lesson, reference and standalone live-function notebooks."""
import ast,base64,hashlib,io,json,re,zlib
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l143';S='0143-relgnn-reproduction';TITLE='RelGNN reproduction: defend the evidence'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
core=(P/'relkit/relgnn_l143.py').read_text();functions=defs(P/'relkit/reproduction_l143.py');checks=defs(P/'_check_l143.py')
summary=json.loads((E/'training.json').read_text())
results='| Evidence lane | Validation MAE | Test MAE | What was executed |\n|---|---:|---:|---|\n'
results+='| Published Table 2 | — | 3.798 | Reported five-seed mean |\n'
results+=f"| Checkpoint-compatible replay | {summary['replay']['val']['mae']:.6f} | {summary['replay']['test']['mae']:.6f} | One released set of weights; reconstructed qualifying types |\n"
results+=f"| Fresh reconstruction, five seeds | {summary['reconstruction']['val']['mean']:.6f} ± {summary['reconstruction']['val']['sample_sd']:.6f} | {summary['reconstruction']['test']['mean']:.6f} ± {summary['reconstruction']['test']['sample_sd']:.6f} | Ten full epochs per seed; sample SD |\n"
results+=f"\nAll **{summary['verified_predictions']:,} held-out predictions** independently aligned and rescored. Fresh score comparison: **{summary['reconstruction']['test']['descriptive_closeness']}**. Historical training: **NOT_ESTABLISHED**. Whole paper: **NOT_RUN**.\n"
audit=json.loads((E/'label-audit.json').read_text());diagnosis=json.loads((E/'diagnosis.json').read_text())
results+=f"\nIndependent raw-table reconstruction checked **{audit['independently_rebuilt_labels']:,} labels**. The separate real first-batch source audit matched **{sum(diagnosis['matched_nonfinite_gradients'].values()):,} nonfinite gradient entries** and a maximum finite-entry difference of **{diagnosis['max_finite_gradient_error']:.2g}**. These gradient counts describe that diagnostic batch, not every seed or batch.\n"
captions={'contract':'Freeze the claim before observing scores. A complete task run and historical identity require different evidence.','compatibility':'Checkpoint shape is insufficient: compare ordered column moments. A supported reconstruction does not identify the historical cache.','architecture':'Full selected RelGNN computation: owner-safe temporal sampling, row/time encoders, route fusion, destination attention and scalar driver prediction.','results':'Measured five-seed test scores and sample SD, separate from checkpoint replay and the published mean.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l143/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l143/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on a narrow screen.</figcaption></figure>')
 for marker,id,fallback in [('SELECTION_WIDGET','l143-selection','Baseline validation [4,3,3]: first minimum selects epoch2.'),('VERDICT_WIDGET','l143-verdict','Baseline five complete fresh fits, illustrative mean3.90: COMPLETE, CLOSE; historical identity NOT_ESTABLISHED.')]:s=s.replace('[['+marker+']]',fallback if portable else f'<div class="route-widget" id="{id}"></div><noscript>{fallback}</noscript>')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write your EXIT explanation before consulting the reference answer.' if portable else '<div id="l143-teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','reproduction-audit'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 143 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/atomic-route.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0142-many-to-many-edge-pathology.html">Lesson 142</a></nav><header><p class="route-kicker">Year 4 · Quarter 3 · Lesson 143</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference="""## Reproduction contract
One full rel-f1/driver-position task; paper Table2 MAE3.798. Complete7453/499/760queries. Fresh reconstruction: five seeds0–4, ten epochs, Adam.005, uniform128/64, first validation minimum. Freeze before test access.

## Compatibility is semantic
qualifying.position is inferred categorical; the checkpoint expects numerical [number,position]. Compare ordered means/population SD to saved buffers. Reconstruct explicitly and preserve the original failure. Matching moments support compatibility; they do not uniquely identify historical preprocessing.

## Three evidence questions
Was every planned fit completed? Is its mean within the frozen descriptive tolerance? Is the historical recipe established? These are separate questions. Replay never substitutes for a fresh seed. Source parity does not prove gradient health.

## Measured evidence
"""+results+"""
## Execution and defense
Default notebook: model fixtures and author-prediction scoring. Explicit full gate: fresh preprocessing, compatible replay, five complete fits. Notebook validation seed100 is separate. Align each prediction by(driver,cutoff); reject missing/duplicate keys. Preserve every failed attempt in the budget. Event-time filtering does not establish feature arrival times. Learner PENDING_WRITTEN_DEFENSE.

[Lesson](../lessons/0143-relgnn-reproduction.html) · [Protocol](../labs/l143-reproduction.md) · [Paper](https://arxiv.org/html/2502.06784v2). Ask the teaching agent to review your evidence verdict.
"""
(R/'reference/relgnn-reproduction.html').write_text(doc('Atomic route reference',reference))
arrays={};histories={}
for phase in ['replay-compatible']+[f'seed-{i}' for i in range(5)]:
 z=np.load(E/phase/'predictions.npz');histories[phase]=json.loads((E/phase/'result.json').read_text())['history']
 for k in z.files:arrays[phase+'_'+k]=z[k]
buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue()
source_files={p.name:p.read_text() for p in (P/'sources/l141').iterdir() if p.is_file()}
packet='''# PROVIDED: author evidence and pinned original sources, not student results.
PACKET=base64.b64decode('''+repr(base64.b64encode(raw).decode())+''')
assert hashlib.sha256(PACKET).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
DATA=np.load(io.BytesIO(PACKET))
SOURCES=json.loads(zlib.decompress(base64.b64decode('''+repr(base64.b64encode(zlib.compress(json.dumps(source_files).encode())).decode())+''')))
SOURCE_ROOT=Path('sources/l141');SOURCE_ROOT.mkdir(parents=True,exist_ok=True)
for name,text in SOURCES.items():(SOURCE_ROOT/name).write_text(text)
AUTHOR_SUMMARY='''+repr(summary)+'''
AUTHOR_HISTORIES='''+repr(histories)+'\nAUTHOR_COMPATIBILITY='+repr(json.loads((E/'compatible/prepared.json').read_text())['compatibility'])+'\n'
bootstrap='''# @colab-bootstrap — default is a portable CPU mechanism and evidence audit.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','sentence-transformers==3.3.1'])
from pathlib import Path
import base64,hashlib,io,json,zlib
import pandas as pd
from IPython.display import display
RUN_FULL_REPRODUCTION=False
FULL_SEEDS=list(range(5));PREPARED_ROOT=None
'''
imports=core.split('def get_atomic_routes',1)[0]
visible=core[core.index('def get_atomic_routes'):]
parity=(P/'_parity_l143.py').read_text().split("if __name__=='__main__':",1)[0].replace('    from relkit.relgnn_l143 import RelGNNConv\n','')
fixture=(P/'_fixture_l143.py').read_text().split("if __name__=='__main__':",1)[0]
trainer='\n'.join(line for line in (P/'_full_l143.py').read_text().splitlines() if line.strip() not in ['from relkit.relgnn_l143 import RelGNN_Model,get_atomic_routes,keyed_mae','from _parity_l143 import original_modules','from relkit.reproduction_l143 import first_validation_min','from relbench.modeling.graph import make_pkey_fkey_graph,get_node_train_table_input'])+'\n'
ast.parse(trainer)
compat=(P/'_compat_l143.py').read_text().split("if __name__",1)[0] if (P/'_compat_l143.py').exists() else ''
compat=compat.replace('from _full_l143 import sha,full_run\n','').replace('from relkit.reproduction_l143 import verify_numeric_layout\n','')
scoring='''# CHECK: your function evaluates all real author prediction vectors.
rows=[];rng=np.random.default_rng(141)
for phase in ['replay-compatible']+[f'seed-{i}' for i in range(5)]:
    for split in ['val','test']:
        prefix=phase+'_'+split+'_';keys=list(zip(DATA[prefix+'entity'],DATA[prefix+'time']))
        order=rng.permutation(len(keys))
        score=keyed_mae(keys,DATA[prefix+'target'],[keys[i] for i in order],DATA[prefix+'pred'][order])
        expected=float(np.mean(np.abs(DATA[prefix+'target']-DATA[prefix+'pred'])))
        assert abs(score-expected)<1e-12
        rows.append(dict(lane=phase,split=split,mae=score,queries=len(keys)))
display(pd.DataFrame(rows))
for phase,history in AUTHOR_HISTORIES.items():
    if history:
        assert len(history)==10 and all(x['queries']==7453 for x in history)
        assert first_validation_min(history)==1+int(np.argmin([x['val_mae'] for x in history]))
records=[dict(seed=i,kind='RECONSTRUCTED_TRAINING',epochs=len(AUTHOR_HISTORIES[f'seed-{i}']),complete=True,test_mae=next(r['mae'] for r in rows if r['lane']==f'seed-{i}' and r['split']=='test')) for i in range(5)]
assert evidence_verdict(records)==AUTHOR_SUMMARY['verdict']
e=AUTHOR_COMPATIBILITY['evidence'];moments={c:(e['mean']['table'][i],e['std']['table'][i]) for i,c in enumerate(['number','position'])}
assert verify_numeric_layout(['number','position'],moments,e['mean']['checkpoint'],e['std']['checkpoint'])
assert sum(r['queries'] for r in rows)==AUTHOR_SUMMARY['verified_predictions']
'''
gate='''# PROVIDED: expensive full-data execution is explicit and separate from the default lane.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for name,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(name).split('+')[0]==version,(name,'requires pinned runtime')
    assert torch.cuda.is_available(),'Pinned GPU environment required'
    prepared=Path(PREPARED_ROOT) if PREPARED_ROOT else Path('l143-prepared')
    if PREPARED_ROOT is None:materialize(prepared,SOURCE_ROOT)
    replay_root=Path('l143-compatible')
    prepare_checkpoint_compatibility(prepared,replay_root)
    replay_result=full_run(Path('l143-runs/replay-compatible'),replay_root,SOURCE_ROOT,replay=True)
    fresh_results=[full_run(Path('l143-runs')/f'seed-{seed}',prepared,SOURCE_ROOT,seed) for seed in FULL_SEEDS]
    print('Full execution complete; historical training remains NOT_ESTABLISHED')
else:print('Default lane: full fresh training NOT_RUN; author evidence independently rescored.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 143 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Tier B: full released relational task. Three live functions guard compatibility, checkpoint selection and verdict. Default CPU execution is distinct from the explicit GPU reproduction gate.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(imports),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  task=None
  if section.startswith('## 2'):task=('verify_numeric_layout','check_layout')
  if section.startswith('## 4'):task=('first_validation_min','check_selection')
  if section.startswith('## 5'):task=('evidence_verdict','check_verdict')
  if task:
   name,check=task;code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
   cells.append(nb.v4.new_code_cell('# TODO — this function is live in the implementation.\n'+code))
   cells.append(nb.v4.new_code_cell('# CHECK — run unchanged.\nimport copy\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 3'):
   for part in re.split(r'(?=^# %% )',visible,flags=re.M):
    if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED — inspect this stage before running.\n'+part))
   cells.append(nb.v4.new_code_cell('# PROVIDED/CHECK: full model fixture and original operator differential test\n'+fixture+'\n'+parity+'\nfixture_report=neural_fixture(RelGNN_Model,get_atomic_routes)\nparity_report=check(SOURCE_ROOT)\nprint(fixture_report);print(parity_report)'))
  if section.startswith('## 6'):cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## Post-EXIT: full-data reproduction code\n\nThe complete preprocessing and trainer are visible below. They preserve live learner functions. The original source is only the differential oracle. Fresh training uses the frozen course recipe, and the compatibility helper documents the released checkpoint type mismatch. The GPU gate is off by default. See the protocol for budget and pinned execution commands.'))
 cells.append(nb.v4.new_code_cell('# PROVIDED: released graph construction and query loader inputs\n'+(P/'sources/l141/relbench__modeling__graph.py').read_text()))
 for part in re.split(r'(?=^def (?:sha|materialize|full_run)\()',trainer,flags=re.M):
  if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED: frozen full-data trainer\n'+part))
 if compat:cells.append(nb.v4.new_code_cell('# PROVIDED: explicit checkpoint compatibility reconstruction\n'+compat))
 cells.append(nb.v4.new_code_cell(gate))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',predictions=sum(r['queries'] for r in rows),fixture=fixture_report,operator=parity_report,full_training='RUN' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l143-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 for i,c in enumerate(cells):c.id=f'l143-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 # Preserve executed output only when the complete source/metadata input is unchanged.
 if solution and path.exists():
  old=nb.read(path,4)
  notebook.metadata=old.metadata
  for new,prev in zip(notebook.cells,old.cells):
   if new.cell_type==prev.cell_type=='code' and new.source==prev.source:new.outputs=prev.outputs;new.execution_count=prev.execution_count;new.metadata=prev.metadata
 nb.write(notebook,path)
print('Built lesson, reference and both standalone notebooks')

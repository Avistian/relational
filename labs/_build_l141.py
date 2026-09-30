"""Canonical lesson, reference and standalone live-function notebooks."""
import ast,base64,hashlib,io,json,re,zlib
from pathlib import Path
import numpy as np
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l141';S='0141-composite-message-passing';TITLE='Composite message passing: trace an atomic route'
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
core=(P/'relkit/relgnn_l141.py').read_text();functions=defs(P/'relkit/relgnn_l141.py');checks=defs(P/'_check_l141.py')
summary=json.loads((E/'training.json').read_text())
results='| Evidence lane | Validation MAE | Test MAE | What was executed |\n|---|---:|---:|---|\n'
results+='| Published Table 2 | — | 3.798 | Reported five-seed mean |\n'
results+=f"| Checkpoint-compatible replay | {summary['replay']['val']['mae']:.6f} | {summary['replay']['test']['mae']:.6f} | One released set of weights; reconstructed qualifying types |\n"
results+=f"| Fresh reconstruction, five seeds | {summary['reconstruction']['val']['mean']:.6f} ± {summary['reconstruction']['val']['sample_sd']:.6f} | {summary['reconstruction']['test']['mean']:.6f} ± {summary['reconstruction']['test']['sample_sd']:.6f} | Ten full epochs per seed; sample SD |\n"
results+=f"\nAll **{summary['verified_predictions']:,} held-out predictions** independently aligned and rescored. Fresh score comparison: **{summary['reconstruction']['test']['descriptive_closeness']}**. Historical training: **NOT_ESTABLISHED**. Whole paper: **NOT_RUN**.\n"
captions={'routes':'A three-key result row yields six ordered composite routes. The pictured arrows show information flow; foreign key declarations point from result to each endpoint.','attention':'Synthetic scalar calculation: constructor [2,4] plus result [1,0] gives fused [3,4]. Query 1 yields 3.731; query 0 yields 3.500. These are not measured embeddings.','architecture':'Selected F1 model: two sampling hops, one composite layer, four 128-channel heads, route sums and a scalar driver head. The source/fact/driver route is highlighted inside the complete forward pass.','results':'Measured fresh seed points and mean ± sample SD; paper mean and checkpoint-compatible replay are separate evidence populations. No causal model comparison is established.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l141/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l141/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on a narrow screen.</figcaption></figure>')
 for marker,id,fallback in [('ROUTES_WIDGET','l141-routes','Baseline: three foreign key roles → six ordered composite routes; one key → two direct routes.'),('ATTENTION_WIDGET','l141-attention','Scalar baseline q=1: scores[3,4], weights[0.269,0.731], output3.731. Change q to0: output3.500.')]:s=s.replace('[['+marker+']]',fallback if portable else f'<div class="route-widget" id="{id}"></div><noscript>{fallback}</noscript>')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write your EXIT explanation before consulting the reference answer.' if portable else '<div id="l141-teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=render(body).replace('<table>','<div class="route-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','atomic-route-viz'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 141 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/atomic-route.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0140-rdl-reproduction-checkpoint.html">Lesson 140</a></nav><header><p class="route-kicker">Year 4 · Quarter 3 · Lesson 141</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Build the route
One foreign key role → direct route in each direction. k≥2 roles → k(k−1) ordered source→fact→destination routes. Count FK roles, not stored reverse edges or unique destination table names. A composite update uses two graph edges; sampler depth differs from model depth.

## Compute the selected operator
1. Fuse `u = W_source sum(source neighbors) + b_source + W_fact fact` separately per ordered route.
2. Project destination to queries and u to keys/values. Scores are dot products divided by √128.
3. Softmax over incoming edges for each destination and head. Four heads each output128channels.
4. Concatenate512channels, add destination skip, project to128. Sum route outputs. Also accumulate fused intermediates at fact nodes.
5. Per-node LayerNorm, ReLU, root selection and scalar head. Train unclipped L1; evaluate with train-target2nd/98th percentile clipping.

## Evidence and safe claims
Released checkpoint requires a compatibility reconstruction for qualifying column types; historical feature semantics remain unestablished. Fresh five-seed training uses documented course choices. Nonfinite numerical-encoder gradients match the released model; finite scores do not prove healthy optimization. Align by unique(entity,cutoff), not position. Source parity, descriptive closeness, temporal event legality and historical identity are different claims.

## Measured evidence
'''+results+'''\n[Lesson](../lessons/0141-composite-message-passing.html) · [Protocol](../labs/l141-reproduction.md) · [RelGNN paper](https://arxiv.org/html/2502.06784v2).
'''
(R/'reference/composite-message-passing.html').write_text(doc('Atomic route reference',reference))
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
AUTHOR_HISTORIES='''+repr(histories)+'\n'
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
visible=core
for name in ['get_atomic_routes','destination_softmax','keyed_mae']:visible=visible.replace(functions[name],'')
visible=visible[visible.index('# %% Composite operator'):]
parity=(P/'_parity_l141.py').read_text().split("if __name__=='__main__':",1)[0].replace('    from relkit.relgnn_l141 import RelGNNConv\n','')
fixture=(P/'_fixture_l141.py').read_text().split("if __name__=='__main__':",1)[0]
trainer='\n'.join(line for line in (P/'_full_l141.py').read_text().splitlines() if line.strip() not in ['from relkit.relgnn_l141 import RelGNN_Model,get_atomic_routes,keyed_mae','from _parity_l141 import original_modules','from relbench.modeling.graph import make_pkey_fkey_graph,get_node_train_table_input'])+'\n'
ast.parse(trainer)
compat=(P/'_compat_l141.py').read_text().split("if __name__",1)[0] if (P/'_compat_l141.py').exists() else ''
compat=compat.replace('from _full_l141 import sha,full_run\n','')
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
    if history:assert len(history)==10 and all(x['queries']==7453 for x in history)
assert sum(r['queries'] for r in rows)==AUTHOR_SUMMARY['verified_predictions']
'''
gate='''# PROVIDED: expensive full-data execution is explicit and separate from the default lane.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for name,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(name).split('+')[0]==version,(name,'requires pinned runtime')
    assert torch.cuda.is_available(),'Pinned GPU environment required'
    prepared=Path(PREPARED_ROOT) if PREPARED_ROOT else Path('l141-prepared')
    if PREPARED_ROOT is None:materialize(prepared,SOURCE_ROOT)
    replay_root=Path('l141-compatible')
    prepare_checkpoint_compatibility(prepared,replay_root)
    replay_result=full_run(Path('l141-runs/replay-compatible'),replay_root,SOURCE_ROOT,replay=True)
    fresh_results=[full_run(Path('l141-runs')/f'seed-{seed}',prepared,SOURCE_ROOT,seed) for seed in FULL_SEEDS]
    print('Full execution complete; historical training remains NOT_ESTABLISHED')
else:print('Default lane: full fresh training NOT_RUN; author evidence independently rescored.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 141 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Tier B: full released relational task. Three live functions feed the actual model and scorer. Default CPU execution is distinct from the explicit GPU reproduction gate.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(imports),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  task=None
  if section.startswith('## 2'):task=('get_atomic_routes','check_routes')
  if section.startswith('## 4'):task=('destination_softmax','check_softmax')
  if section.startswith('## 6'):task=('keyed_mae','check_mae')
  if task:
   name,check=task;code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
   cells.append(nb.v4.new_code_cell('# TODO — this function is live in the implementation.\n'+code))
   cells.append(nb.v4.new_code_cell('# CHECK — run unchanged.\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 5'):
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
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',predictions=sum(r['queries'] for r in rows),fixture=fixture_report,operator=parity_report,full_training='RUN' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l141-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 for i,c in enumerate(cells):c.id=f'l141-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 # Preserve executed output only when the complete source/metadata input is unchanged.
 if solution and path.exists():
  old=nb.read(path,4)
  if [c.source for c in old.cells]==[c.source for c in cells]:notebook=old
 nb.write(notebook,path)
print('Built lesson, reference and both standalone notebooks')

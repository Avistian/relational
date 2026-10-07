"""Canonical content, portable inline student/solution notebooks and reference."""
import ast,base64,hashlib,io,json,re,zlib
from pathlib import Path
import nbformat as nb,numpy as np
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l142';S='0142-many-to-many-edge-pathology';TITLE='Many-to-many edges: trace where the signal mixes'
summary=json.loads((E/'training.json').read_text())
results='| Evidence | Validation MAE | Test MAE | Parameters |\n|---|---:|---:|---:|\n| Published RelGNN five-seed mean | — | 3.798 | — |\n'
for arm,label in [('composite','Fresh composite, five seeds'),('ordinary','Fresh ordinary, five seeds')]:
 v=summary['arms'][arm]['val'];t=summary['arms'][arm]['test'];results+=f"| {label} | {v['mean']:.6f} ± {v['sample_sd']:.6f} | {t['mean']:.6f} ± {t['sample_sd']:.6f} | {summary['parameter_counts'][arm]:,} |\n"
g=summary['paired']['test'];results+=f"\nMean paired test gap (ordinary − composite): **{g['mean']:+.6f} ± {g['sample_sd']:.6f} MAE** (sample SD across five seed pairs). Composite versus paper: **{summary['closeness']}**, using the frozen ±0.20 descriptive band. **{summary['verified_predictions']:,} held-out predictions** independently aligned and rescored. Historical identity remains **NOT_ESTABLISHED**.\n"
results+='\nComposite wins **three of five test seed pairs**, while ordinary attention has the lower mean validation MAE. The sign describes these frozen runs only. Neither a win nor a loss identifies the bridge mechanism as the cause; the comparison also changes capacity and normalization depth.\n'
captions={'walks':'Exact linear diagnostic: one source path, two fact paths and two destination paths. Identity weights and self sums differ from the trained attention network.','collision':'Swapping source and third-role values preserves the shared scalar sum. It destroys a source-specific distinction but preserves their total.','architecture':'Both full networks reach two graph edges. Composite uses one route layer; ordinary uses two edge-attention layers. Shared input design does not imply equal capacity.','results':'Measured test MAE for five complete fits per arm. Seed labels connect the comparison; positive ordinary-minus-composite gaps favor composite.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l142/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l142/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption} Scroll horizontally on narrow screens.</figcaption></figure>')
 for marker,id,fallback in [('WALKS_WIDGET','l142-walks','Baseline s=2,f=1,d=3: ordinary10, composite6. Destination-only input1 produces ordinary2, composite1.'),('HUB_WIDGET','l142-hub','Baseline s=2,n=8,f=1,d=3: ordinary18, composite6. Swap s/n: ordinary18, composite12.')]:
  s=s.replace('[['+marker+']]',fallback if portable else f'<div class="path-widget" id="{id}"></div><noscript>{fallback}</noscript>')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write your defense before consulting the reference.' if portable else '<div id="l142-teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','pathology-viz'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 142 — '+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/atomic-route.css"><link rel="stylesheet" href="../assets/pathology.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0141-composite-message-passing.html">Lesson 141</a> · <a href="../labs/html/'+S+'.html">Executed lab</a></nav><header><p class="route-kicker">Year 4 · Quarter 3 · Lesson 142</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
ref='''## Trace first
For the scalar synchronous self-plus-neighbor update on s↔f↔d with third hub neighbor n: f₁=f+s+d+n, d₁=d+f, d₂=s+2f+2d+n. Destination echo: d→f→d. Repeated fact: f→f→d and f→d→d. Composite diagnostic: d′=d+f+s. These are specified identity-weight examples, not universal learned-network coefficients.

## Distinction, collision, counterexample
With f=1,d=3, (s,n)=(2,8) and (8,2) both give ordinary18; source-route outputs6and12. Harmful if target depends on s; harmless if it depends only on s+n. Ordinary relation-specific channels can preserve roles; composite route sums can also collide. Do not erase fact attributes, event multiplicity or timestamps.

## Full comparison
One composite layer versus two ordinary edge-attention layers; same two-edge reach and input/training protocol. Four heads of128channels, concat512, project128. Different capacity, normalization depth and RNG trajectories. No claim that this reproduces the paper's baseline or isolates one cause. Each sampled row obeys its owning query cutoff.

## Evidence
'''+results+'''\nNonfinite gradients remain a released-encoder limitation. Five seed-pair gaps are descriptive; repeated query rows are not independent trials. Default notebook replay is not fresh training or learner mastery. Historical identity NOT_ESTABLISHED; whole paper NOT_RUN.

## Defense checklist
Derive coefficients; state update assumptions; give a harmful collision and a harmless one; identify what the paired experiment holds fixed; name an unresolved confound. Ask the teaching agent to review your derivation.

[Lesson](../lessons/0142-many-to-many-edge-pathology.html) · [Protocol](../labs/l142-reproduction.md) · [RelGNN motivation](https://arxiv.org/html/2502.06784v2#S3).
'''
(R/'reference/many-to-many-edge-pathology.html').write_text(doc('Many-to-many pathology reference',ref))
def definitions(path):
 text=path.read_text();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
functions=definitions(P/'relkit/pathology_l142.py');checks=definitions(P/'_check_l142.py')
arrays={};histories={}
for arm in ['composite','ordinary']:
 for seed in range(5):
  phase=f'{arm}-{seed}';z=np.load(E/phase/'predictions.npz');histories[phase]=json.loads((E/phase/'result.json').read_text())['history']
  for k in z.files:arrays[phase+'_'+k]=z[k]
buf=io.BytesIO();np.savez_compressed(buf,**arrays);raw=buf.getvalue()
sources={p.name:p.read_text() for p in (P/'sources/l141').iterdir() if p.is_file()}
packet="PACKET=base64.b64decode("+repr(base64.b64encode(raw).decode())+")\nassert hashlib.sha256(PACKET).hexdigest()=="+repr(hashlib.sha256(raw).hexdigest())+"\nDATA=np.load(io.BytesIO(PACKET))\nSOURCES=json.loads(zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(json.dumps(sources).encode())).decode())+")))\nSOURCE_ROOT=Path('sources/l141');SOURCE_ROOT.mkdir(parents=True,exist_ok=True)\nfor name,text in SOURCES.items():(SOURCE_ROOT/name).write_text(text)\nAUTHOR_SUMMARY="+repr(summary)+"\nAUTHOR_HISTORIES="+repr(histories)+"\n"
bootstrap='''# @colab-bootstrap — full GPU execution is separate from default CPU checks.
import os,sys,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','pytorch-frame==0.2.3','torch-geometric==2.6.1','relbench==1.1.0','sentence-transformers==3.3.1'])
from pathlib import Path
import base64,hashlib,io,json,zlib
import numpy as np,pandas as pd,torch
from IPython.display import display
RUN_FULL_REPRODUCTION=False
FULL_SEEDS=list(range(5));PREPARED_ROOT=None
'''
core=(P/'relkit/relgnn_l142.py').read_text().replace('from relkit.pathology_l142 import edge_sum,route_fuse\n','')
fixture=(P/'_fixture_l141.py').read_text().split("if __name__=='__main__':")[0]
fixture+='''\ndef ordinary_factory(*args,**kwargs):
    kwargs.update(dict(zip(['data','col_stats_dict','num_model_layers','channels','out_channels','aggr','norm'],args)))
    return OrdinaryModel(**kwargs)
fixture_report={name:neural_fixture(model,get_atomic_routes) for name,model in [('composite',RelGNN_Model),('ordinary',ordinary_factory)]}
print(fixture_report)
'''
parity=(P/'_parity_l141.py').read_text().split("if __name__=='__main__':")[0].replace('    from relkit.relgnn_l141 import RelGNNConv\n','')
parity+='\nparity_report=check(SOURCE_ROOT)\nprint(parity_report)\n'
trainer='\n'.join(line for line in (P/'_full_l142.py').read_text().splitlines() if not line.strip().startswith(('from relkit.relgnn_l142 import','from _parity_l141 import','from relbench.modeling.graph import')))+'\n'
scoring='''# CHECK: use your keyed function on all primary author predictions.
rows=[];rng=np.random.default_rng(142)
for split in ['val','test']:
    gaps=[]
    for seed in range(5):
        prefix=lambda arm:f'{arm}-{seed}_{split}_'
        ka=list(zip(DATA[prefix('composite')+'entity'],DATA[prefix('composite')+'time']))
        kb=list(zip(DATA[prefix('ordinary')+'entity'],DATA[prefix('ordinary')+'time']))
        y=DATA[prefix('composite')+'target'];np.testing.assert_array_equal(y,DATA[prefix('ordinary')+'target'])
        order=rng.permutation(len(kb))
        gap=paired_loss_gap(ka,y,ka,DATA[prefix('composite')+'pred'],[kb[i] for i in order],DATA[prefix('ordinary')+'pred'][order])
        gaps.append(float(gap.mean()))
        for arm in ['composite','ordinary']:
            score=float(np.abs(y-DATA[prefix(arm)+'pred']).mean());rows.append(dict(arm=arm,seed=seed,split=split,queries=len(y),mae=score))
    np.testing.assert_allclose(gaps,AUTHOR_SUMMARY['paired'][split]['seed_gaps'],atol=1e-12)
for arm in ['composite','ordinary']:
    for split in ['val','test']:
        assert abs(np.mean([r['mae'] for r in rows if r['arm']==arm and r['split']==split])-AUTHOR_SUMMARY['arms'][arm][split]['mean'])<1e-12
for history in AUTHOR_HISTORIES.values():assert len(history)==10 and all(x['queries']==7453 and x['steps']==15 for x in history)
assert sum(r['queries'] for r in rows)==AUTHOR_SUMMARY['verified_predictions']
display(pd.DataFrame(rows))
'''
gate='''# Full named task and course comparison; explicit GPU execution.
if RUN_FULL_REPRODUCTION:
    import importlib.metadata as md
    for name,version in {'torch':'2.5.1','torch-geometric':'2.6.1','pytorch-frame':'0.2.3','relbench':'1.1.0','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
        assert md.version(name).split('+')[0]==version,(name,'requires pinned runtime')
    assert torch.cuda.is_available(),'Pinned GPU environment required'
    prepared=Path(PREPARED_ROOT) if PREPARED_ROOT else Path('l142-prepared')
    if PREPARED_ROOT is None:materialize(prepared,SOURCE_ROOT)
    fresh_results=[full_run(Path('l142-runs')/f'{arm}-{seed}',prepared,SOURCE_ROOT,seed,arm=arm) for arm in ['composite','ordinary'] for seed in FULL_SEEDS]
    print('Full selected execution complete; historical training NOT_ESTABLISHED')
else:print('Default lane: full fresh training NOT_RUN; author files independently rescored.')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 142 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. All three learner functions are live in neural computation or the measured paired analysis. Default CPU evidence replay and explicit GPU training are separate.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(packet,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  if section.startswith('## 3'):
   for name,checkname in [('edge_sum','check_edge_sum'),('route_fuse','check_fuse')]:
    code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.append(nb.v4.new_code_cell('# TODO: live neural operator\n'+code));cells.append(nb.v4.new_code_cell('# CHECK: independent arithmetic\n'+checks[checkname]+'\n'+checkname+'('+name+')'))
  if section.startswith('## 4'):
   for part in re.split(r'(?=^# %% )',core,flags=re.M):
    if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED: visible complete model stage\n'+part))
   cells.append(nb.v4.new_code_cell('# CHECK: both full neural models forward/backward\n'+fixture))
   cells.append(nb.v4.new_code_cell('# CHECK: independent original composite outputs/gradients\n'+parity))
  if section.startswith('## 5'):
   name='paired_loss_gap';code=functions[name] if solution else functions[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: paired_loss_gap")'
   cells.append(nb.v4.new_code_cell('# TODO: live paired measurement\n'+code));cells.append(nb.v4.new_code_cell(checks['check_pair']+'\ncheck_pair(paired_loss_gap)'));cells.append(nb.v4.new_code_cell(scoring))
 cells.append(nb.v4.new_markdown_cell('## Full execution code\n\nThe preprocessing and trainer below run complete task queries. The gate is off by default. Source files embedded above are differential references. Model, learner functions and trainer remain visible and executable inline. Use a fresh output directory and the pinned environment; see the protocol for the aggregate budget.'))
 cells.append(nb.v4.new_code_cell('# PROVIDED: source graph construction and query inputs\n'+(P/'sources/l141/relbench__modeling__graph.py').read_text()))
 for part in re.split(r'(?=^def (?:sha|materialize|full_run)\()',trainer,flags=re.M):
  if part.strip():cells.append(nb.v4.new_code_cell('# PROVIDED: full frozen trainer\n'+part))
 cells.append(nb.v4.new_code_cell(gate))
 cells.append(nb.v4.new_code_cell("report=dict(status='PASS',predictions=sum(r['queries'] for r in rows),fixtures=fixture_report,operator=parity_report,full_training='RUN' if RUN_FULL_REPRODUCTION else 'NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')\nPath('l142-report.json').write_text(json.dumps(report,indent=2))\nprint(report)"))
 for i,c in enumerate(cells):c.id=f'l142-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nb.read(path,4)
  for new_cell,old_cell in zip(notebook.cells,old.cells):
   if new_cell.cell_type==old_cell.cell_type=='code' and new_cell.source==old_cell.source:
    new_cell.outputs=old_cell.outputs;new_cell.execution_count=old_cell.execution_count
 nb.write(notebook,path)
print('Built lesson, reference and portable notebooks')

"""Deterministic connected lesson, reference and portable full audit notebooks."""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l156';F=P/'figures/l156';S='0156-temporal-leakage-audit';TITLE='Temporal leakage audit: trace every information path'
r=json.loads((E/'report.json').read_text());m=json.loads((E/'input-manifest.json').read_text());pre=json.loads((E/'preflight.json').read_text())
table='| Lane | Validation MAE | Test MAE | Strict fit/availability verdict |\n|---|---:|---:|---|\n'
for name,l in r['lanes'].items():
 v=l['metrics']['val'];t=l['metrics']['test'];table+=f"| {'Released source' if name=='paper' else 'Fit by 2005 correction'} | {v['mean']:.4f} ± {v['sd']:.4f} | {t['mean']:.4f} ± {t['sd']:.4f} | {l['strict_policy_verdict']['status']} |\n"
result=f"**Measured consequence:** corrected minus released test MAE is **{r['corrected_minus_released_mae']['test']:+.4f} finishing positions**. Positive means the corrected lane is worse. The temporal policy is justified by its information contract, not by a preferred score. The released mean's frozen ±0.20 comparison with Table 7 is **{r['reference_paper_band']['test']}**; the corrected lane is not that published experiment."
coverage=r['lanes']['paper']['coverage'];c2=r['lanes']['fit_horizon']['coverage'];gradient=r['lanes']['paper']['first_backward_nonfinite']
evidence=f"**Measured coverage:** all 8,712 labels, {r['sql_values']:,} independently reconstructed SQL values, and {r['predictions']:,} held-out predictions across ten primary fits. Each lane audits {coverage['queries']:,} yielded query occurrences. The released lane checks {coverage['dated_node_occurrences']:,} dated-node occurrences and {coverage['edge_occurrences']:,} edge occurrences; these are repeated sampled occurrences, not unique database rows. Equality-boundary occurrences: {coverage['equality_nodes']:,}. Nonfinite first-backward entries per released seed: {list(gradient.values())}. [Machine-readable coverage](../labs/evidence/l156/report.json)."
portfolio='**Portfolio coverage:** F1 receives the new full census and fresh instrumented fits. The classification entry has existing completed fits, but a fresh L156 execution of its new auditor is **NOT_RUN**. Recommendation retains its earlier 32-batch validation pilot: full reproduction **INCOMPLETE**, test **NOT_RUN**. Reviewing their pinned contracts does not close those gaps. [Coverage record](../labs/evidence/l156/portfolio-review.json).'
captions={'clocks':'Synthetic event 9/arrival 12 at query cutoff 10: event time passes while availability fails. Query B cannot lend its cutoff 20.','window':'Synthetic label trace: include times 11 and 15 in (10,15], exclude 10 and 16, mean(2+6)/2=4.','paths':'Actual released and corrected information paths. Preprocessing population and owner-cutoff sampling are separate checks.','results':'Actual primary five-seed runs per policy. Points are seeds; diamonds and bars are mean and sample seed SD. Separate validation/test scales.'}
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 for key,value in [('TABLE',table),('RESULT',result),('EVIDENCE',evidence),('PORTFOLIO',portfolio)]:s=s.replace('[['+key+']]',value)
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((F/(name+'.png')).read_bytes()).decode() if portable else '../labs/figures/l156/'+name+'.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="route-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for marker,web,plain in [('WARMUP','<div id="warmup"></div>','Recall without notes: what owns a cutoff? When can a future outcome be a legal training label?'),('PREDICT','<div id="predict"></div>','Predict: does a passing event-time check prove historical availability? Explain before proceeding.'),('WIDGET','<div id="l156-audit" class="temporal-audit"></div><noscript>For cutoff 10: event9/arrival9passes; event 9/arrival 12fails; event9/unknown arrival is NOT_ESTABLISHED. A schedule for event12published9can pass.</noscript>','Work through the CHECK fixtures: cutoff 10,event9,arrival12fails. Future schedule12published9passes. Unknown arrival prevents historical sign-off.'),('TEACHBACK','<div id="teachback"></div>','Write your defense before comparing against the teaching agent\'s feedback.')]:s=s.replace('[['+marker+']]',plain if portable else web)
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,body,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="route-scroll" tabindex="0"><table\1>',render(body)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','predict','teachback','temporal-audit','l156-lesson'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson156 — '+title+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','atomic-route','checkpoint','temporal-audit'])+'</head><body class="checkpoint"><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0155-compare-manual-fe.html">Lesson155</a></nav><header><p class="route-kicker">Year4 · Quarter4 · Lesson156</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
reference='''## Reusable REG audit checklist

For each check record policy,coverage,evidence identifier,status and limitation. Required checks: query identity,label windows/cohort,sampling,SQL history,processor fitting,availability,model selection,prediction keys.

| Contract | Test | Boundary |
|---|---|---|
| Event input | event < cutoff or event ≤ cutoff,declared explicitly | Owner cutoff at every hop |
| Availability | observed publication/ingestion/version time ≤ cutoff | None means unknown |
| Schedule | publication ≤ cutoff; event may be later | Do not invent publication history |
| Label | cutoff < event ≤ cutoff+horizon | Check population and label maturity |
| Processor fit | Type inference,vocabulary and stats use allowed fit population | Transform later rows without refitting |
| Graph identity | Original row/edge IDs; no cross-query edges | Sampled census is not all possible draws |
| Selection | Freeze by validation,then score test | Prior test exposure remains disclosed |
| Sign-off | Known FAIL dominates; missing checks NOT_CHECKED; missing history NOT_ESTABLISHED | A hash attests bytes,not independent review |

## Correction workflow
Preserve released baseline. Name the violated policy. Freeze correction before inspecting scores. Refit the complete affected pipeline,align all predictions,report before/after and remaining gaps. Never choose temporal correctness by test score.

## Current measured audit
'''+table+'\n\n'+result+'\n\n'+portfolio+'''

[Lesson](../lessons/0156-temporal-leakage-audit.html) · [Notebook](../labs/0156-temporal-leakage-audit.ipynb) · [Protocol](../labs/l156-reproduction.md) · [Report](../labs/evidence/l156/report.md) · [Primary source](https://arxiv.org/html/2407.20060v1#S2).

Ask the teaching agent to review your signed-scope statement. Author checks do not establish learner mastery; PENDING_WRITTEN_DEFENSE.
'''
(R/'reference/reg-temporal-leakage-audit.html').write_text(doc('Temporal audit reference',reference))
def defs(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
fns=defs(P/'relkit/temporal_audit_l156.py');checks=defs(P/'_check_l156.py');replay=defs(P/'_replay_l156.py')
packet={name:base64.b64encode((E/name).read_bytes()).decode() for name in m['files']};packed=base64.b64encode(zlib.compress(json.dumps(packet,sort_keys=True).encode(),9)).decode()
bootstrap='''# @colab-bootstrap: default offline full-data audit; paid training is OFF.
import os,sys,ast,base64,copy,hashlib,json,math,statistics,tempfile,zlib
from pathlib import Path
import numpy as np
from IPython.display import display,Markdown
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
RUN_FULL_REPRODUCTION=False
'''
transport='''# PROVIDED: frozen primary AUTHOR evidence. No fresh fits in the default lane.
MANIFEST='''+repr(m)+'''
PACKET='''+repr(packed)+'''
assert hashlib.sha256(PACKET.encode()).hexdigest()=='''+repr(hashlib.sha256(packed.encode()).hexdigest())+'''
workspace=tempfile.TemporaryDirectory(prefix='l156-audit-');REPLAY_ROOT=Path(workspace.name)/'evidence';REPLAY_ROOT.mkdir()
CODE_ROOT=Path(workspace.name)/'code';(CODE_ROOT/'relkit').mkdir(parents=True)
for name,encoded in json.loads(zlib.decompress(base64.b64decode(PACKET))).items():
 dest=(REPLAY_ROOT/name).resolve();assert dest.is_relative_to(REPLAY_ROOT.resolve())
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==MANIFEST['files'][name]
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
'''
# Visible code files are written in an isolated temporary directory. They are
# imports only when full gate is enabled; default replay needs only numpy/IPython.
module_paths=['relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/regression_l152.py','_run_l117.py','_run_l152.py','_run_l156.py']
vendor={str(p.relative_to(P)):p.read_text() for p in (P/'sources/l117').iterdir() if p.is_file()}
vendor['requirements-l117-runtime.txt']=(P/'requirements-l117-runtime.txt').read_text()
source_transport='''# PROVIDED: unchanged licensed upstream sources for original-model parity.
VENDOR='''+repr(vendor)+'''
for name,text in VENDOR.items():
 dest=CODE_ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
'''
full_gate='''# PROVIDED: complete selected computational experiment, not a smoke test.
# Use the pinned GPU recipe in modal/l156_notebook_check.py. No automatic install.
if RUN_FULL_REPRODUCTION:
 import importlib,importlib.metadata as md
 for name,version in {'torch':'2.5.1','pytorch-frame':'0.2.3','relbench':'1.1.0','torch-geometric':'2.6.1','numpy':'1.26.4','pandas':'2.2.3','sentence-transformers':'3.3.1'}.items():
  assert md.version(name).split('+')[0]==version,(name,'Use the pinned runtime')
 import torch
 assert torch.cuda.is_available(),'Use the pinned GPU runtime'
 # The learner function remains live inside every fresh training audit.
 import types
 pkg=types.ModuleType('relkit');pkg.__path__=[str(CODE_ROOT/'relkit')];sys.modules['relkit']=pkg
 live=types.ModuleType('relkit.temporal_audit_l156');live.audit_observations=audit_observations
 sys.modules[live.__name__]=live
 sys.path.insert(0,str(CODE_ROOT));runner=importlib.import_module('_run_l156')
 full_root=Path('l156-full');full_root.mkdir(exist_ok=False);fresh=[]
 for lane in ['released','fit_horizon']:
  for seed in range(5):
   output=full_root/lane/f'seed-{seed}';result=runner.run(seed,10,output,lane)
   arr=np.load(output/'predictions.npz')
   for split in ['val','test']:
    expected=np.load(REPLAY_ROOT/f'{split}-labels.npz');truth={(int(e),int(t)*10**9):float(y) for e,t,y in zip(expected['entity'],expected['time'],expected['target'])}
    keys=list(zip(arr[split+'_entity'].tolist(),arr[split+'_time'].tolist()))
    assert len(set(keys))==len(keys) and set(keys)==set(truth)
    target=np.array([truth[k] for k in keys]);np.testing.assert_allclose(target,arr[split+'_target'],rtol=0,atol=1e-12)
    assert abs(float(np.abs(arr[split+'_pred']-target).mean())-result['scores'][split])<1e-12
   fresh.append(dict(lane=lane,seed=seed,epochs=10,scores=result['scores']))
 # Actual processor invariance: later numerical values must not alter fit stats.
 import pandas as pd
 from relbench.base import Database,Table
 from torch_frame import stype
 from torch_frame.config import TextEmbedderConfig
 def example(value):
  frame=pd.DataFrame({'id':[0,1,2],'date':pd.to_datetime(['2000-01-01','2001-01-01','2008-01-01']),'value':[1.,3.,value]})
  return Database({'events':Table(frame,pkey_col='id',fkey_col_to_pkey_table={},time_col='date')})
 kinds={'events':{'value':stype.numerical}}
 _,s1=runner.fit_horizon_graph(example(1000.),copy.deepcopy(kinds),None)
 _,s2=runner.fit_horizon_graph(example(999999.),copy.deepcopy(kinds),None)
 assert str(s1)==str(s2),'Post-horizon perturbation changed fitted stats'
 _,b1=runner.BASE_GRAPH(example(1000.),copy.deepcopy(kinds),None)
 _,b2=runner.BASE_GRAPH(example(999999.),copy.deepcopy(kinds),None)
 assert str(b1)!=str(b2),'Control perturbation failed to change released stats'
 Path('l156-full-report.json').write_text(json.dumps(dict(status='PASS',fits=10,predictions=12590,records=fresh,processor_invariance='PASS',primary_mean_inclusion=False),indent=2))
 print('Ten full notebook fits PASS; excluded from primary means')
'''
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson156 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · PROVIDED / TODO / CHECK / EXIT. Default: offline replay of complete real F1 audit evidence, requiring numpy and IPython. Implement three functions; their outputs determine the real report. The optional full gate retrains both five-seed lanes using the pinned GPU environment; it downloads the hash-checked archives and pinned text model. Gates do not impose a dollar cap; author dispatch separately enforces USD10. The source-code cells write only into a fresh temporary directory.'),nb.v4.new_code_cell(bootstrap),nb.v4.new_code_cell(transport,metadata={'tags':['data-payload']})]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for num,name,check in [('2','audit_observations','check_observations'),('3','audit_label_windows','check_labels'),('6','audit_verdict','check_verdict')]:
   if section.startswith('## '+num):
    code=fns[name] if solution else fns[name].split('\n',1)[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.append(nb.v4.new_code_cell('# TODO: this function is used by the full evidence replay.\n'+code))
    cells.append(nb.v4.new_code_cell('# CHECK: reject wrong boundaries and unsupported certainty.\n'+checks['rejects']+'\n'+checks['observations']+'\n'+checks[check]+'\n'+check+'('+name+')\nprint("PASS '+name+'")'))
  if section.startswith('## 6'):
   cells.append(nb.v4.new_markdown_cell('### PROVIDED · Independent replay\n\nThe visible code verifies input hashes, reconstructs all labels and actual SQL dependencies, aligns held-out predictions, checks all training epochs and validation selections, and calls your sign-off function. A human signature is not fabricated.'))
   for name in ['verify_inputs','replay','render_report']:cells.append(nb.v4.new_code_cell(replay[name]))
   cells.append(nb.v4.new_code_cell('report=replay(REPLAY_ROOT,MANIFEST,observations=audit_observations,labels=audit_label_windows,verdict=audit_verdict)\nassert report["predictions"]==12590\ndisplay(Markdown(render_report(report)))'))
 cells.append(nb.v4.new_code_cell('''# EXIT: attach your scoped defense; mastery requires review.
WRITTEN_DEFENSE=""
Path('l156-audit-report.json').write_text(json.dumps(dict(report,written_defense=WRITTEN_DEFENSE),indent=2))
Path('l156-audit-report.md').write_text(render_report(report)+'\\n'+WRITTEN_DEFENSE)
Path('l156-report.json').write_text(json.dumps(dict(status='PASS',predictions=report['predictions'],labels=sum(report['label_counts'].values()),learner='PENDING_WRITTEN_DEFENSE'),indent=2))
'''))
 cells.append(nb.v4.new_markdown_cell('## Optional full reproduction · visible implementation\n\nUse Python3.11, Torch2.5.1/CUDA12.4, Frame0.2.3, PyG2.6.1, pyg-lib0.4.0 and the remaining exact requirements in the transported file. The [Modal recipe](https://avistian.github.io/relational/modal/l156_notebook_check.py) specifies the environment. Set `RUN_FULL_REPRODUCTION=True` in the bootstrap cell only in that runtime, then run all. These code cells write the visible implementation into an isolated directory. They do not execute training until the final gate. Full model, graph construction, trainer, original identity auditor and new correction are shown below. Only original licensed reference sources are transported as data.'))
 cells.append(nb.v4.new_code_cell(source_transport,metadata={'tags':['data-payload']}))
 for path in module_paths:
  source=(P/path).read_text()
  parts=re.split(r'(?=^# %%)',source,flags=re.M) if 'rdl_l117.py' in path else [source]
  for i,part in enumerate(parts):
   if not part:continue
   cells.append(nb.v4.new_markdown_cell('### PROVIDED · '+path+(' · part'+str(i+1) if len(parts)>1 else '')+'\n\nThis is executable source, shared with the author run.'))
   cells.append(nb.v4.new_code_cell('%%writefile '+('-a ' if i else '')+'{CODE_ROOT}/'+path+'\n'+part))
 cells.append(nb.v4.new_code_cell(full_gate))
 cells.append(nb.v4.new_code_cell("workspace.cleanup()\nprint('Audit notebook complete; live Colab NOT_CHECKED; defense PENDING_WRITTEN_DEFENSE')"))
 for i,c in enumerate(cells):c.id=f'l156-{i:03d}'
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}})
 path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nb.read(path,4);notebook.metadata=old.metadata
  previous=[c for c in old.cells if c.cell_type=='code'];current=[c for c in notebook.cells if c.cell_type=='code']
  if [c.source for c in previous]==[c.source for c in current]:
   for prev,new in zip(previous,current):
    new.outputs=prev.outputs;new.execution_count=prev.execution_count;new.metadata=prev.metadata
 nb.write(notebook,path)
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
executed=nb.read(P/'solutions'/(S+'.ipynb'),4)
if all(c.execution_count is not None for c in executed.cells if c.cell_type=='code'):
 exporter=HTMLExporter(template_name='lab');exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={'data-payload'}),enabled=True)
 html,_=exporter.from_notebook_node(executed);(P/'html'/(S+'.html')).write_text(html)
print('Built L156 lesson,reference,student and solution')

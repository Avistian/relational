"""Canonical lesson plus portable live analysis and complete separately gated trainers."""
import ast,base64,gzip,hashlib,io,json,re,textwrap,zipfile
from pathlib import Path
import nbformat as nb
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l137';S='0137-error-analysis-reg';TITLE='Error analysis on REG: where does the GNN lose?'
def definitions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
funcs=definitions(P/'relkit/error_reg_l137.py');checks=definitions(P/'_check_l137.py');summary=json.loads((E/'errors.json').read_text());frozen=json.loads((E/'frozen.json').read_text());training=json.loads((E/'training.json').read_text())
result='| Split | GNN MAE ± seed SD | FE MAE ± seed SD | GNN − FE |\n|---|---:|---:|---:|\n'
for split,r in summary['splits'].items():result+=f"| {split} | {r['gnn_mean']:.4f} ± {r['gnn_sd']:.4f} | {r['fe_mean']:.4f} ± {r['fe_sd']:.4f} | {r['slices']['all']['mean']:+.4f} |\n"
result+='\nThese use all five fresh runs and all held-out queries. The paired scorer uses the same archived target precision for both methods. No ensemble prediction or selected best seed is substituted.'
t=summary['splits']['test'];v=summary['splits']['val'];chosen=frozen['selected'];ci=t['selected']['interval'];finding=f"**The frozen nomination is `{chosen}`:** more than 22 recorded prior results. Its validation gap is **{v['selected']['mean']:+.4f}**, on {v['selected']['rows']} queries / {v['selected']['drivers']} drivers. Test gives **{t['selected']['mean']:+.4f}**, with descriptive 95% interval **[{ci['low']:+.4f}, {ci['high']:+.4f}]**, on {t['selected']['rows']} queries / {t['selected']['drivers']} drivers. The interval spans zero. The observed-recent-slots slice has **362 validation rows but zero test rows**; it cannot validate a test-time failure claim."
traintext='**Selected GNN paper target:** RelBench v1 Table 7, rel-f1/driver-position, basic RDL. Five seeds, ten full epochs each.\n\n| Split | Fresh mean | Paper target | Predeclared descriptive tolerance | Verdict |\n|---|---:|---:|---:|---|\n'
for split,m in training['metrics'].items():traintext+=f"| {split} | {m['mean']:.6f} | {m['paper_target']:.3f} | ±0.2 MAE | {m['verdict']} |\n"
traintext+='\nCLOSE is a descriptive score check, not historical identity or statistical equivalence. FE completed all 50 search trials and five selected refits. Independent checks reconstruct every SQL feature, all saved tree predictions, and all GNN query identities.'
captions={'paired':'Synthetic two-query example: opposing paired losses cancel in the mean.','snapshot':'Illustrative frozen snapshot: recorded recency grows while newer real events are unavailable.','clusters':'Synthetic driver bootstrap: duplicate whole clusters and retain pooled query weights.','seeds':'Measured five-run MAE; matching run labels do not imply common random numbers.','slices':'Measured all planned slices; intervals condition on fitted models and split. Unsupported groups receive no interval.'}
def prose(portable=False,student=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',result).replace('[[FINDING]]',finding).replace('[[TRAINING]]',traintext)
 for marker,name in [('PAIR_CODE','paired_errors'),('NOMINATE_CODE','nominate_slice'),('CLUSTER_CODE','cluster_interval')]:s=s.replace('[['+marker+']]', 'Implement this contract in the live TODO below.' if student else '```python\n'+funcs[name]+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l137/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l137/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 for marker,id,fallback in [('PAIR_WIDGET','l137-pair','Baseline FE14 → mean gap0. FE10 → mean gap2: GNN loses.'),('CLUSTER_WIDGET','l137-cluster','Baseline A+B → 6/3=2. A+A → 0/4=0. B+B → 12/2=6.')]:s=s.replace('[['+marker+']]',fallback if portable else f'<div class="tuning-control" id="{id}"></div><noscript>{fallback}</noscript>')
 s=s.replace('[[WARMUP]]','' if portable else '<div id="warmup"></div>').replace('[[TEACHBACK]]','Write your defense before asking the teacher to evaluate it.' if portable else '<div id="l137-teachback"></div>')
 if portable:s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def doc(title,s,interactive=False):
 body=re.sub(r'<table([^>]*)>',r'<div class="stream-scroll" tabindex="0"><table\1>',render(s)).replace('</table>','</table></div>')
 scripts=['retrieval-pool','retrieval-bank','teachback','paired-reg-errors'] if interactive else []
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/tuning-budget.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0136-leaderboard-literacy.html">Lesson 136</a></nav><header><p class="stream-kicker">Year 4 · Quarter 2 · Lesson 137</p><h1>'+title+'</h1></header>'+body+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(doc(TITLE,prose(),True))
ref='''## Query-paired diagnosis

Join on entity and cutoff, require a bijection, then compute |GNN − target| − |FE − target|. Positive means the GNN loses. Average per-run losses, not predictions, unless evaluating an ensemble is explicitly intended.

## Slice protocol

Define feature-only rules first. Fit thresholds on training features. Require ≥30 queries and ≥10 drivers. On validation nominate the largest positive supported gap; tie-break by name. Freeze before test slice analysis. Report all declared slices and their support. An empty slice has no estimate, not zero error. Previously used test data remain reused.

## Uncertainty

Resample whole drivers with replacement, retain their query rows and recompute the pooled mean. Driver means must be weighted by resampled query counts. State that intervals condition on the chosen split and fitted models; race/time dependence remains. Selection, overlapping slices and missing test support constrain claims.

## Repair contract

Observation → competing explanations → one intervention with other settings fixed → validation selection → new held-out temporal evaluation. An association alone does not identify a causal mechanism.

[Lesson](../lessons/0137-error-analysis-reg.html) · [Protocol](../labs/l137-reproduction.md) · [RelBench v1](https://arxiv.org/html/2407.20060v1)
'''
(R/'reference/error-analysis-reg.html').write_text(doc('REG error-analysis field guide',ref))
def packed(raw):return repr(base64.b64encode(gzip.compress(raw,mtime=0)).decode())
raw=(E/'portable.json').read_bytes()
replay='''# PROVIDED: full primary predictions, keys and feature-only masks, hash checked.
raw=gzip.decompress(base64.b64decode('''+packed(raw)+'''))
assert hashlib.sha256(raw).hexdigest()=='''+repr(hashlib.sha256(raw).hexdigest())+'''
DATA=json.loads(raw)
EXPECTED='''+repr(summary)+'''
FROZEN='''+repr(frozen['selected'])+'''
# Validation first, using your live implementations.
v=DATA['val'];ids=np.array([k[0] for k in v['keys']])
vd=np.mean([paired_errors(v['keys'],v['target'],v['keys'],g,v['keys'],f) for g,f in zip(v['gnn'],v['fe'])],axis=0)
masks={k:np.array(m,dtype=bool) for k,m in v['masks'].items() if k!='all'}
nomination=nominate_slice(vd,ids,masks);assert nomination['selected']==FROZEN
rows=[];count=0
for split,x in DATA.items():
 ids=np.array([k[0] for k in x['keys']]);d=np.mean([paired_errors(x['keys'],x['target'],x['keys'],g,x['keys'],f) for g,f in zip(x['gnn'],x['fe'])],axis=0)
 for name,mask in x['masks'].items():
  mask=np.array(mask,bool);expected=EXPECTED['splits'][split]['slices'][name]
  assert int(mask.sum())==expected['rows']
  mean=float(d[mask].mean()) if mask.any() else None
  if mean is not None:assert abs(mean-expected['mean'])<1e-12
  interval=cluster_interval(d[mask],ids[mask]) if expected['supported'] else None
  if interval:assert interval==expected['interval']
  rows.append(dict(split=split,slice=name,rows=int(mask.sum()),drivers=len(np.unique(ids[mask])),gap=mean,low=interval['low'] if interval else None,high=interval['high'] if interval else None))
 count+=len(x['keys'])*5*2
display(pd.DataFrame(rows))
report=dict(status='PASS',individual_predictions=count,selected=nomination['selected'],slices=len(rows),learner='PENDING_WRITTEN_DEFENSE')
Path('l137-report.json').write_text(json.dumps(report,indent=2));print(report)
'''
# Archive only the source inputs needed by the FE gate, without hidden model code.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for rel in ['db.zip','driver-position.zip','data-manifest.json','inferred_stypes.py','f1/driver-position/feats.sql']:
  info=zipfile.ZipInfo(rel,date_time=(2026,9,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,(P/'sources/l129'/rel).read_bytes())
archive=buf.getvalue()
fe_funcs=(P/'relkit/fe_experiment_l129.py').read_text().replace('    from relkit.manual_fe_l129 import choose_trial\n','')
fe_helpers=definitions(P/'relkit/manual_fe_l129.py')
fe_gate='''if RUN_FULL_FE_REPRODUCTION:
 import importlib.metadata as md
 for package,version in {'torch':'2.2.2','pytorch-frame':'0.2.2','lightgbm':'4.3.0','optuna':'3.6.1','duckdb':'0.10.3','pandas':'2.0.3','numpy':'1.26.0','jinja2':'3.1.3'}.items():
  assert md.version(package)==version, (package,'use requirements-l129-runtime.txt')
'''+textwrap.indent(fe_funcs+'\n'+fe_helpers['choose_trial']+'\n'+fe_helpers['align_predictions'],' ')+'\n'
fe_gate+='''if RUN_FULL_FE_REPRODUCTION:
 import zipfile,importlib.util
 raw=gzip.decompress(base64.b64decode('''+packed(archive)+'''))
 assert hashlib.sha256(raw).hexdigest()=='''+repr(hashlib.sha256(archive).hexdigest())+'''
 source=Path('l137-fe-source');source.mkdir(exist_ok=True)
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for name in z.namelist():
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
 tables,queries,_=load_archives(source)
 features=make_features(tables,queries,(source/'f1/driver-position/feats.sql').read_text())
 spec=importlib.util.spec_from_file_location('types',source/'inferred_stypes.py');types=importlib.util.module_from_spec(spec);spec.loader.exec_module(types)
 dataset,frames,matrices=materialize_features(features,types.task_to_stypes['rel-f1-driver-position'])
 output=Path('l137-fe-full');output.mkdir(exist_ok=False);records=[]
 for seed in range(5):
  model,trace,chosen=tune_fe(matrices,seed,num_trials=10,rounds=2000,threads=4)
  model.save_model(str(output/f'seed-{seed}.txt'));scores={};saved={}
  for split in ['val','test']:
   q=queries[split];df=features[split];keys=list(zip(q.driverId,q.date))
   pred=np.array(align_predictions(list(zip(df.driverId,df.date)),model.predict(matrices[split][0]),keys));target=q.position.to_numpy()
   scores[split]=float(np.mean(np.abs(pred-target)))
   saved.update({split+'_pred':pred,split+'_target':target,split+'_entity':q.driverId.to_numpy(),split+'_time':q.date.astype('int64').to_numpy()})
  np.savez_compressed(output/f'seed-{seed}.npz',**saved)
  record=dict(seed=seed,trace=trace,selected_trial=chosen,scores=scores);records.append(record)
  (output/f'seed-{seed}.json').write_text(json.dumps(record,indent=2))
 (output/'packet.json').write_text(json.dumps(dict(status='PASS',fits=5,trials=50,records=records),indent=2))
 print('Full FE reproduction:50 trials and5 refits completed')
'''
# Prior standalone notebook exposes the complete inherited GNN in coherent chunks.
prior=nb.read(P/'solutions/0136-leaderboard-literacy.ipynb',4)
start=next(i for i,c in enumerate(prior.cells) if c.cell_type=='markdown' and c.source.startswith('## PROVIDED · Full selected RDL'))
gnn_cells=[]
for c in prior.cells[start:]:
 if c.cell_type=='markdown':gnn_cells.append(nb.v4.new_markdown_cell(c.source.replace('this lesson','this reproduction appendix').replace('Your `regression_score` checks every resulting prediction.', 'The independent MAE calculation checks every resulting prediction.')))
 else:
  s=c.source.replace('l136','l137').replace('RUN_FULL_REPRODUCTION = False','RUN_FULL_REPRODUCTION = RUN_FULL_GNN_REPRODUCTION').replace("score=regression_score(a[split+'_target'],a[split+'_pred'],1.)['mae']","score=float(np.mean(np.abs(a[split+'_target']-a[split+'_pred'])))")
  gnn_cells.append(nb.v4.new_code_cell('if RUN_FULL_GNN_REPRODUCTION:\n'+textwrap.indent(s,'    ')))
for solution in [False,True]:
 cells=[nb.v4.new_markdown_cell('# Lesson 137 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · PROVIDED scaffolding → three live TODOs → CHECKs → real full-data analysis → written EXIT. Default execution uses saved fresh author predictions; full FE and GNN training have separate gates and pinned environments.'),nb.v4.new_code_cell('''# @colab-bootstrap: default analysis needs only NumPy, pandas and IPython.
import importlib.util,subprocess,sys
missing=[p for p in ['numpy','pandas','IPython'] if importlib.util.find_spec(p) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import base64,gzip,hashlib,io,json,math
from pathlib import Path
import numpy as np,pandas as pd
from IPython.display import display
'''+checks['rejects'])]
 for section in re.split(r'(?=^## )',prose(True,not solution),flags=re.M):
  if not section.strip():continue
  cells.append(nb.v4.new_markdown_cell(section))
  for prefix,name,check in [('## 2','paired_errors','check_pair'),('## 3','nominate_slice','check_nominate'),('## 4','cluster_interval','check_cluster')]:
   if section.startswith(prefix):
    code=funcs[name] if solution else funcs[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
    cells.extend([nb.v4.new_code_cell(code),nb.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")')])
  if section.startswith('## 5'):cells.extend([nb.v4.new_markdown_cell('### CHECK · Your functions on every primary held-out prediction\n\nCompressed data below are checksum-verified author evidence. Changing your live pairing, nomination or resampling function changes this analysis; no hidden result function substitutes for it.'),nb.v4.new_code_cell(replay)])
 cells.extend([nb.v4.new_markdown_cell('## Optional complete training lanes\n\nBoth are OFF. Run FE in the pinned CPU runtime; run GNN in its separate pinned GPU runtime. They intentionally use different Frame/Torch versions. Never silently combine them into a new protocol. Dollar and timeout enforcement is in the CLI/Modal operators, not these notebook switches. Default analysis requires neither torch nor GPU.'),nb.v4.new_code_cell('RUN_FULL_FE_REPRODUCTION = False\nRUN_FULL_GNN_REPRODUCTION = False')])
 cells.append(nb.v4.new_markdown_cell('### PROVIDED · Released SQL and full FE pipeline\n\nOriginal user-study commit445bb7a3b1230f49f8e5890ae81754d3e365680f. SQL and schema are supplied verbatim with attribution; no new license is asserted. Frame search code is MIT. Full query rows, train-fitted mappings, ten-trial search and selected refit remain visible.\n\n```sql\n'+(P/'sources/l129/f1/driver-position/feats.sql').read_text()+'\n```'))
 cells.append(nb.v4.new_code_cell(fe_gate))
 cells.append(nb.v4.new_code_cell('if RUN_FULL_GNN_REPRODUCTION:\n    import torch\n    torch.set_num_threads(1)'))
 cells.extend(gnn_cells)
 notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
 dest=P/('solutions' if solution else '')/f'{S}.ipynb'
 if dest.exists():
  old=nb.read(dest,4);oc=[c for c in old.cells if c.cell_type=='code'];nc=[c for c in notebook.cells if c.cell_type=='code']
  if [c.source for c in oc]==[c.source for c in nc]:
   notebook.metadata=old.metadata
   for a,b in zip(nc,oc):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 for i,c in enumerate(notebook.cells):c.id=f'l137-{i:03d}'
 nb.write(notebook,dest)
print('Built L137 lesson, reference and standalone student/solution notebooks')

"""Build aligned lesson, reference, and standalone notebooks with complete inputs."""
import ast,base64,hashlib,io,json,re,textwrap,zipfile
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0129-manual-feature-engineering';TITLE='Manual feature engineering: what RDL must beat'
O=P/'evidence/l129';summary=json.loads((O/'summary.json').read_text());audit=json.loads((O/'sql-audit.json').read_text());trace=json.loads((O/'prediction-trace.json').read_text())
def definitions(path):
 s=path.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
funcs=definitions(P/'relkit/manual_fe_l129.py');checks=definitions(P/'_check_l129.py');trainer=(P/'relkit/fe_experiment_l129.py').read_text();portable_trainer='\n'.join(line for line in trainer.splitlines() if 'from relkit.manual_fe_l129 import' not in line)
results='**Five complete fresh searches, ten trials each.**\n\n| Split | Manual FE mean MAE | Sample SD | L127 RDL mean MAE | FE − RDL |\n|---|---:|---:|---:|---:|\n'
for s in ['val','test']:
 m=summary['metrics'][s];c=summary['comparison'][s];results+=f"| {s} | {m['mean']:.6f} | {m['sample_sd']:.6f} | {c['rdl_mean']:.6f} | {c['fe_minus_rdl']:+.6f} |\n"
results+='\n| Search seed | Selected trial | Trees | Validation MAE | Test MAE |\n|---|---:|---:|---:|---:|\n'
for row in summary['seeds']:results+=f"| {row['seed']} | {row['selected_trial']} | {row['trees']} | {row['val']:.6f} | {row['test']:.6f} |\n"
results+=f"\nAll **{summary['predictions']:,} predictions** independently checked. Five searches and their selected refits took **{summary['training_seconds']:.2f} seconds** inside the local worker processes; pilot, startup, preparation and audits are separate. **Paid cloud spend: USD 0.** Seed SD is descriptive, not a confidence interval."
v=audit['example']['features'];real=f"For driver **{audit['example']['entity']}** at **{audit['example']['cutoff'][:10]}**, driver points **{v['driver_points']:g}** and constructor points **{v['constructor_points']:g}** give **{v['points_ratio']:.2f}**. Driver standing **{v['driver_position']:g}** minus constructor standing **{v['constructor_position']:g}** gives **{v['position_diff']:g}**. Past slot 2 has a recorded grid but missing finishing position: missing values need not mean the whole row is absent."
pt=f"In seed 0, this row follows **{len(trace['first_tree_path'])}** tests in the first tree and reaches leaf **{trace['first_tree_leaf']:.6f}**. The first ten trees sum to **{trace['first_ten_sum']:.6f}**; the remaining **{trace['tree_count']-10}** contribute **{trace['remaining_sum']:.6f}**. Total prediction **{trace['prediction']:.6f}**, target **{trace['target']:.1f}**, absolute error **{abs(trace['prediction']-trace['target']):.6f}**. The error is large on this row even though the model's aggregate MAE is lower. [Machine-readable trace](../labs/evidence/l129/prediction-trace.json)."
captions={'workflow':'The target stays outside features. Explicit relational SQL yields 50 features plus retained numeric driverId; train-fitted mappings feed ten-trial selection and keyed test scoring.', 'features':'Actual driver7 at2009-08-08: points ratio18/40=.45, standing difference9−3=6. Recent race slots use global race IDs.', 'trees':'Actual seed0 first-tree decisions and full ensemble sum. Leaf contributions already include scaling; do not multiply by the learning rate again.', 'scores':'Five complete FE searches compared descriptively with existing L127 RDL. Detail scales; bars show sample seed SD, not confidence intervals.', 'effort':'Original human work, automated replay runtime and a new learner effort log measure different activities.'}
fallback='Static cutoff trace: at day10 with an8-day lookback, event/arrival pairs(5,5) and(9,10) yield values4 and2: count2,mean3,recency1. Ignoring arrival admits99 and changes mean to35. Including the event at cutoff admits88 and changes mean to31.333.'
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results).replace('[[REAL_TRACE]]',real).replace('[[PREDICTION_TRACE]]',pt)
 snippet=definitions(P/'relkit/fe_experiment_l129.py')['tune_fe'];snippet='\n'.join(x for x in snippet.splitlines() if 'from relkit' not in x)
 s=s.replace('[[TRAIN_CODE]]','```python\n'+snippet+'\n```')
 for name,caption in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l129/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l129/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
 s=s.replace('[[CUTOFF_WIDGET]]',fallback if portable else '<div id="l129-cutoff"></div><noscript><p>'+fallback+'</p></noscript>')
 s=s.replace('[[WARMUP]]','Recall query identity and temporal visibility before reading.' if portable else '<div id="warmup"></div>')
 s=s.replace('[[TEACHBACK]]','Write your answer about human effort before checking the evidence.' if portable else '<div id="l129-teachback"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,text,interactive=False):
 body=render(text).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','event-snapshot','reproduction','task-table','rdl-stack-viz'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0128-task-taxonomy.html">Lesson128</a></nav><header><p class="stream-kicker">Year4 · Quarter1 · Lesson129</p><h1>'+title+'</h1></header>'+body+'</article>'+(''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','teachback','fe-cutoff-viz','l129-lesson']) if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
# Deterministic embedded payload; no hidden solution functions in student assets.
paths=[P/'sources/l129'/x for x in ['db.zip','driver-position.zip','data-manifest.json','inferred_stypes.py','f1/driver-position/feats.sql']]+[O/'matrices.npz',O/'summary.json']
for seed in range(5):paths.extend(O/f'paper/seed-{seed}'/name for name in ['model.txt','predictions.npz','result.json'])
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for path in paths:
  info=zipfile.ZipInfo(str(path.relative_to(P)),date_time=(2026,9,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
raw=buf.getvalue();encoded=base64.b64encode(raw).decode();digest=hashlib.sha256(raw).hexdigest()
bootstrap='''# @colab-bootstrap: only install missing packages for default teaching/replay.
import importlib.util,subprocess,sys
packages={'numpy':'numpy','pandas':'pandas','pyarrow':'pyarrow','sklearn':'scikit-learn','lightgbm':'lightgbm','duckdb':'duckdb','jinja2':'jinja2'}
missing=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import io,json,base64,hashlib,math,zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
'''
unpack='''# PROVIDED: full archives, matrices, five models and every held-out prediction.
raw=base64.b64decode('''+repr(encoded)+''')
assert hashlib.sha256(raw).hexdigest()=='''+repr(digest)+'''
root=Path('l129-portable');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 for name in z.namelist():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
source=root/'sources/l129';evidence=root/'evidence/l129'
print('Verified complete portable bundle:',len(raw),'bytes')
'''
course='''# YOUR past_summary constructs all8,712 real-query inputs for a small course tree.
from sklearn.tree import DecisionTreeRegressor
start=time.perf_counter()
with zipfile.ZipFile(source/'db.zip') as z:
 events_df=pd.read_parquet(io.BytesIO(z.read('db/results.parquet')))
events_df=events_df[(events_df.date<=pd.Timestamp('2010-01-01'))].drop_duplicates(['raceId','driverId'])
event_groups={}
for row in events_df.itertuples():
 day=pd.Timestamp(row.date).value/86400e9
 event_groups.setdefault(int(row.driverId),[]).append(dict(entity=int(row.driverId),event=day,available=day,value=float(row.positionOrder)))
labels={}
with zipfile.ZipFile(source/'driver-position.zip') as z:
 for split in ['train','val','test']:labels[split]=pd.read_parquet(io.BytesIO(z.read('driver-position/'+split+'.parquet')))
fill_mean=float(labels['train'].position.median());X={};checked=0
for split,frame in labels.items():
 rows=[]
 for q in frame.itertuples():
  t=pd.Timestamp(q.date).value/86400e9;events=event_groups.get(int(q.driverId),[])
  got=past_summary(events,int(q.driverId),t,60)
  # Independently vectorized expected contract on real history.
  days=np.array([e['event'] for e in events]);values=np.array([e['value'] for e in events]);mask=(days>t-60)&(days<t)
  assert got['count']==int(mask.sum())
  if mask.any():
   assert abs(got['mean']-float(values[mask].mean()))<1e-10
   assert abs(got['days_since_latest']-(t-days[mask].max()))<1e-10
  else:assert got['mean'] is None and got['days_since_latest'] is None
  rows.append([got['count'],fill_mean if got['mean'] is None else got['mean'],61 if got['days_since_latest'] is None else got['days_since_latest']]);checked+=1
 X[split]=np.array(rows)
course_model=DecisionTreeRegressor(max_depth=3,random_state=129)
course_model.fit(X['train'],labels['train'].position)
course_scores={s:float(np.mean(np.abs(course_model.predict(X[s])-labels[s].position))) for s in ['val','test']}
print('COURSE_ONLY:',course_scores,'queries',checked,'seconds',time.perf_counter()-start)
print('Arrival=event is an explicit assumption. This small tree is NOT the released50-feature experiment.')
'''
tree_source=definitions(P/'_analyze_l129.py')['tree_predict']
replay='''# YOUR selection and alignment plus independent saved-tree evaluation.
import lightgbm as lgb,statistics
arrays=np.load(evidence/'matrices.npz',allow_pickle=False)
summary=json.loads((evidence/'summary.json').read_text());scores=[];count=0;tree_error=0
for seed in range(5):
 folder=evidence/f'paper/seed-{seed}';r=json.loads((folder/'result.json').read_text());pred=np.load(folder/'predictions.npz')
 assert choose_trial(r['trace'])==r['selected_trial']
 model=lgb.Booster(model_file=str(folder/'model.txt'));trees=model.dump_model()['tree_info'];row={}
 for split in ['val','test']:
  x=arrays[split+'_x'];raw=sum(tree_predict(t['tree_structure'],x,np.arange(len(x))) for t in trees)
  error=float(np.max(np.abs(raw-model.predict(x,num_threads=2))));assert error<1e-10;tree_error=max(tree_error,error)
  fk=list(zip(arrays[split+'_feature_id'],arrays[split+'_feature_time']));qk=list(zip(arrays[split+'_query_id'],arrays[split+'_query_time']))
  aligned=np.array(align_predictions(fk,raw,qk));np.testing.assert_allclose(aligned,pred[split+'_pred'],rtol=0,atol=1e-10)
  # Deliberately reverse feature order: key-based output must stay identical.
  assert align_predictions(fk[::-1],raw[::-1],qk)==aligned.tolist()
  score=math.fsum(abs(float(y)-float(p)) for y,p in zip(pred[split+'_target'],aligned))/len(aligned)
  assert abs(score-r['scores'][split])<1e-12;row[split]=score;count+=len(aligned)
 scores.append(row);print('Search',seed,'selected',r['selected_trial'],'MAE',row)
aggregate={s:dict(mean=statistics.mean(r[s] for r in scores),sample_sd=statistics.stdev(r[s] for r in scores)) for s in ['val','test']}
for s in aggregate:
 for k in aggregate[s]:assert abs(aggregate[s][k]-summary['metrics'][s][k])<1e-12
report=dict(status='PASS',predictions=count,real_course_queries=checked,course_scores=course_scores,aggregate=aggregate,maximum_tree_error=tree_error,scope='Small new course tree + full author model/evidence replay, not50 new fits',learner_status='PENDING_WRITTEN_DEFENSE')
Path('l129-report.json').write_text(json.dumps(report,indent=2));print(report)
'''
gate='''RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
 # Requires pinned runtime; does not silently install a different model stack.
 import importlib.metadata as md
 for package,version in {'torch':'2.2.2','pytorch-frame':'0.2.2','lightgbm':'4.3.0','optuna':'3.6.1','duckdb':'0.10.3','pandas':'2.0.3','numpy':'1.26.0','jinja2':'3.1.3'}.items():
  assert md.version(package)==version,(package,'use requirements-l129-runtime.txt')
 import importlib.util
 tables,queries,raw_counts=load_archives(source)
 full_features=make_features(tables,queries,(source/'f1/driver-position/feats.sql').read_text())
 spec=importlib.util.spec_from_file_location('stypes',source/'inferred_stypes.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 dataset,frames,matrices=materialize_features(full_features,module.task_to_stypes['rel-f1-driver-position'])
 output=Path('l129-fresh-full');output.mkdir(exist_ok=False)
 for seed in range(5):
  model,trials,chosen=tune_fe(matrices,seed,num_trials=10,rounds=2000,threads=4)
  model.save_model(str(output/f'seed-{seed}.txt'))
  measured={};saved={}
  for split in ['val','test']:
   df=full_features[split];q=queries[split]
   p=np.array(align_predictions(list(zip(df.driverId,df.date)),model.predict(matrices[split][0]),list(zip(q.driverId,q.date))))
   measured[split]=float(np.mean(abs(p-q.position.to_numpy())))
   saved.update({split+'_pred':p,split+'_target':q.position.to_numpy(),split+'_entity':q.driverId.to_numpy(dtype='int64'),split+'_time':q.date.astype('int64').to_numpy()})
  np.savez_compressed(output/f'seed-{seed}.npz',**saved)
  (output/f'seed-{seed}.json').write_text(json.dumps(dict(seed=seed,selected_trial=chosen,trace=trials,scores=measured),indent=2))
  print('FRESH FULL',seed,measured)
else:
 print('Full new50-trial training OFF. Default run is course training + complete author replay.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson129 · '+TITLE+'\n\n'+('Teacher reference' if solution else 'Student lab')+' · Three TODOs control real feature extraction, validation selection and prediction alignment. Default: train one small course tree and replay all author models. Full50-trial training is explicitly gated. PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_code_cell(checks['rejected']))
 for name,check,desc in [('past_summary','check_past','Filter by entity, strict event window and arrival cutoff; return missing for no history.'),('choose_trial','check_select','Choose earliest validation minimum; test scores cannot influence selection.'),('align_predictions','check_align','Validate complete unique query keys, then reorder by(entity,cutoff).')]:
  body=funcs[name] if solution else funcs[name].splitlines()[0]+'\n    raise NotImplementedError("TODO: '+name+'")'
  cells.extend([nbf.v4.new_markdown_cell('## TODO · '+name+'\n\n'+desc),nbf.v4.new_code_cell(body),nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+name+')\nprint("PASS: '+name+'")')])
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · complete pinned inputs\n\nNo repository imports or network downloads for data. The bundle contains all raw tables, all task rows, all feature matrices, five trained models and every held-out prediction.'),nbf.v4.new_code_cell(unpack),nbf.v4.new_markdown_cell('## COURSE · train a tree on your features\n\nUses result positionOrder, strict60-day lookback, assumed arrival=event, frozen2010 database and a fixed depth-three tree. No tuning, no paper-score claim.'),nbf.v4.new_code_cell(course),nbf.v4.new_markdown_cell('## REPLAY · independently traverse every saved tree\n\nProvided tree traversal is distinct from LightGBM prediction; your alignment and selection functions are used on every search.'),nbf.v4.new_code_cell(tree_source),nbf.v4.new_code_cell(replay)])
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · full released SQL\n\nRead every ASOF join and time predicate. Full template, without feature omissions:\n\n```sql\n'+(P/'sources/l129/f1/driver-position/feats.sql').read_text()+'\n```'))
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · complete pipeline implementation\n\nFunctions below perform archive validation, snapshot filtering, SQL extraction, train-fitted materialization and full search. The upstream search class is pinned under sources/l129/frame with MIT attribution. In this portable copy, tune_fe uses your choose_trial from the notebook namespace.'),nbf.v4.new_code_cell(portable_trainer),nbf.v4.new_markdown_cell('## FRESH FULL reproduction · optional gate\n\nOFF by default. Requires pinned runtime from the protocol. Runs all five ten-trial searches, not a smoke test. Local CPU is sufficient; this cell has no monetary enforcement or automatic timeout. Author CLI runs were bounded. Fresh output directory prevents overwriting previous work. Live Colab NOT_CHECKED.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT\n\nSave l129-report.json, your implementations and your completed effort log. Write the five defenses from the lesson. An executed reference notebook does not establish your mastery. PENDING_WRITTEN_DEFENSE.')])
 for i,c in enumerate(cells):c.id=f'l129-{i:03d}'
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
reference='''## A defensible manual-feature baseline

1. Freeze(entity,cutoff), future target, cohort and evaluation split.
2. Write a feature hypothesis before a join.
3. Restrict historical facts by event AND arrival time. Distinguish published future schedules from future outcomes.
4. Preserve missing history and query rows. Fit preprocessing on training rows.
5. Tune with validation only; align predictions by both key fields; score test after selection.
6. Log human work separately from machine execution and reusable infrastructure.

## Released F1 computation

Strict ASOF driver standing → same-race result → constructor standing. Past slots use last_race_id−i+1 and two calendar months; next slots use last_race_id+i and one calendar month. Slots are not personal last/next starts. Ratios use NULLIF for zero denominators. Numerical driverId is retained; date is ignored by the LightGBM adapter. Final input:14 categorical+37 numerical columns. Frozen2010-01-01 database means no recent/upcoming slots for test.

## Model and protocol

L1 boosting; prediction is sum of saved leaf contributions.10 configurations per search;2000-tree cap;50-round validation early stopping; train-only refit of winner. Five explicit TPE seeds0–4; original search unseeded. Standard deviation describes search variation, not a confidence interval.

## Claim boundaries

Full selected released pipeline replay complete on pinned v1 archives; historical data/search/model identity and paper-score parity NOT_ESTABLISHED. Figure3 manualFE is distinct from Table7 raw-entity LightGBM. One original expert's marginal human-effort result cannot be reproduced by timing automated SQL. New effort log remains learner-owned.

[Lesson](../lessons/0129-manual-feature-engineering.html) · [Protocol](../labs/l129-reproduction.md) · [Effort log](../labs/l129-effort-log.md) · [Evidence](../labs/evidence/l129/summary.json) · [Primary paper§6](https://arxiv.org/html/2407.20060v1#S6).
'''
(R/'reference/manual-feature-engineering.html').write_text(document('Manual feature engineering · reference',reference))
print('Built L129 lesson, reference, student/solution; portable bundle bytes',len(raw))

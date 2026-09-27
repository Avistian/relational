"""Rescore complete published regression archives with independent and source oracles."""
import ast,hashlib,io,json,math,types,typing,urllib.request,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from relkit.leaderboard_l136 import align_predictions,regression_score,complete_board
P=Path(__file__).resolve().parent;S=P/'sources/l136';D=P/'results/l136/leaderboard-data';D.mkdir(parents=True,exist_ok=True)
provenance=json.loads((P/'_sources_l136.json').read_text())
for name,row in provenance['files'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==row['sha256'],name
source=(S/'submit.py').read_text();tree=ast.parse(source)
tasks=next(ast.literal_eval(n.value)['regression'] for n in tree.body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id',None)=='LEADERBOARD_TASKS')
ns=dict(pd=pd,np=np,Sequence=typing.Sequence)
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in ['_coerce_keys','_key_set','_validate_keys','_aligned_to_gt']:
  exec('from __future__ import annotations\n'+ast.get_source_segment(source,n),ns)
metrics=types.ModuleType('pinned_metrics');exec(compile((S/'metrics.py').read_text(),str(S/'metrics.py'),'exec'),metrics.__dict__)
stds=json.loads((S/'regression_stds.json').read_text())['stds'];datafiles={};results={};portable={};rows_total=0
entries={393:'kapso',380:'gnn',397:'plurel'}
boards={i:json.loads((S/f'entry-{i}.json').read_text()) for i in entries}
archives={i:zipfile.ZipFile(S/(name+'.zip')) for i,name in entries.items()}
base=f"https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/{provenance['hf_revision']}/"
def fetch(remote):
 path=D/remote;path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  raw=urllib.request.urlopen(base+remote,timeout=90).read();path.write_bytes(raw)
 datafiles[remote]=dict(url=base+remote,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
 return path
for task in tasks:
 dataset,name=task.split('/');root=f'{dataset}/tasks/{name}'
 spec=yaml.safe_load(fetch(root+'/manifest.yaml').read_text());gt=pd.read_parquet(fetch(root+'/test.parquet'))
 keycols=[spec['entity_col'],spec['time_col']];target=spec['target_col']
 assert not gt[keycols].isna().any().any()
 expected=list(gt[keycols].itertuples(index=False,name=None));task_results={}
 # Verify hosted normalization from all train labels, not a convenience subset.
 train=pd.read_parquet(fetch(root+'/train.parquet'));calculated=float(train[target].std(ddof=1));std=stds[task]
 scale_verdict='CLOSE' if math.isclose(calculated,std,rel_tol=2e-6) else 'DIFFERS_FROM_HOSTED_CONSTANT'
 if scale_verdict!='CLOSE':print('Scale audit:',task,calculated,'vs hosted',std,flush=True)
 for issue,z in archives.items():
  csvname=task.replace('/','__')+'.csv';raw=z.read(csvname);df=pd.read_csv(io.BytesIO(raw));df=ns['_coerce_keys'](df,gt,keycols)
  ns['_validate_keys'](df,gt,keycols,target)
  source_pred=ns['_aligned_to_gt'](df,gt,keycols,target).to_numpy()
  supplied=list(df[keycols].itertuples(index=False,name=None))
  pred=np.array(align_predictions(expected,supplied,df[target].tolist()))
  np.testing.assert_array_equal(pred,source_pred)
  scores=regression_score(gt[target].tolist(),pred.tolist(),std)
  oracle=metrics.make_nmae(lambda:std)(gt[target].to_numpy(),source_pred)
  assert abs(scores['nmae']-oracle)<1e-12
  reported=boards[issue]['boards']['regression']['results'][task]
  assert abs(scores['nmae']-reported)<1e-10,(task,issue,scores,reported)
  scores.update(reported=reported,delta=scores['nmae']-reported,source_metric=oracle,prediction_csv_sha256=hashlib.sha256(raw).hexdigest())
  task_results[str(issue)]=scores;rows_total+=len(gt)
  if task=='rel-f1/driver-position':
   # Original-order rows exercise actual keyed alignment in the portable notebook.
   portable[str(issue)]=dict(name=boards[issue]['name'],keys=[[int(a),int(pd.Timestamp(b).value)] for a,b in supplied],pred=df[target].tolist(),reported=reported)
 results[task]=dict(keys=keycols,target=target,rows=len(gt),train_rows=len(train),hosted_train_std=std,recomputed_train_std=calculated,normalization_audit=scale_verdict,entries=task_results)
 if task=='rel-f1/driver-position':
  portable['truth']=dict(keys=[[int(a),int(pd.Timestamp(b).value)] for a,b in expected],target=gt[target].tolist(),train_std=std)
 print(task,len(gt),'rows × 3 entries: exact score replay',flush=True)
aggregates={}
for issue,entry in boards.items():
 values={t:results[t]['entries'][str(issue)]['nmae'] for t in tasks};mean=complete_board(values,tasks);reported=entry['boards']['regression']['mean'];assert abs(mean-reported)<1e-12
 aggregates[str(issue)]=dict(name=entry['name'],mean=mean,reported=reported,delta=mean-reported,tasks=len(tasks))
summary=dict(status='COMPLETE_EVALUATION_REPLAY',canonical_tasks=tasks,entries=aggregates,tasks=results,predictions_rescored=rows_total,data_files=datafiles,source_commit=provenance['relbench_commit'],hf_revision=provenance['hf_revision'],upstream_alignment='EXACT_ALL_ROWS',metric_tolerance=1e-10,training_reproduction='NOT_ESTABLISHED',full_submission_cli='NOT_RUN; exact pinned alignment and metric functions executed',task_config_comparison='Current HF tasks; historical training identity separate')
E=P/'evidence/l136';E.mkdir(parents=True,exist_ok=True)
(E/'leaderboard.json').write_text(json.dumps(summary,indent=2));(E/'portable-f1.json').write_text(json.dumps(portable,separators=(',',':')))
print(json.dumps(aggregates,indent=2));print('Predictions independently rescored:',rows_total)

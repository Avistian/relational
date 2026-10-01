"""Reconstruct all task rows using independent pandas and original pinned SQL."""
import gzip,hashlib,importlib.util,io,json,urllib.request,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from relbench.datasets import get_dataset
from relkit.tasks_l124 import make_labels,validate_task
P=Path(__file__).resolve().parent;S=P/'sources/l152';S.mkdir(parents=True,exist_ok=True)
commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639';manifest={}
for name,path in [('task_f1.py','relbench/tasks/f1.py'),('task_base.py','relbench/base/task_base.py'),('dataset_f1.py','relbench/datasets/f1.py')]:
 url=f'https://raw.githubusercontent.com/snap-stanford/relbench/{commit}/{path}';raw=urllib.request.urlopen(url,timeout=30).read();(S/name).write_bytes(raw);manifest[name]=dict(url=url,sha256=hashlib.sha256(raw).hexdigest())
(S/'LICENSE').write_bytes((P/'sources/l117/LICENSE').read_bytes())
dataset=get_dataset('rel-f1',download=True);db=dataset.get_db(upto_test_timestamp=False)
for filename,digest in [('db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'),('tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e')]:
 assert hashlib.sha256((Path(dataset.cache_dir)/filename).read_bytes()).hexdigest()==digest
spec=importlib.util.spec_from_file_location('task_source_l124',S/'task_f1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);task=module.DriverPositionTask(dataset)
results=db.table_dict['results'].df;ids=dataset.get_db().table_dict['drivers'].df.driverId.tolist()
report={};portable={'events':json.loads(results[['driverId','date','positionOrder']].to_json(orient='records',date_format='iso',double_precision=15)),'entity_ids':ids,'splits':{}}
for split in ['train','val','test']:
 source_db=dataset.get_db(upto_test_timestamp=split!='test');step=pd.Timedelta(days=60)
 if split=='train':times=pd.date_range(dataset.val_timestamp-step,source_db.min_timestamp,freq=-step)
 elif split=='val':times=pd.date_range(dataset.val_timestamp,min(dataset.val_timestamp+39*step,dataset.test_timestamp-step),freq=step)
 else:times=pd.date_range(dataset.test_timestamp,min(dataset.test_timestamp+39*step,source_db.max_timestamp-step),freq=step)
 events=source_db.table_dict['results'].df
 reconstructed=make_labels(events,times)
 reconstructed=reconstructed[reconstructed.driverId.isin(ids)]
 original=task.make_table(source_db,times).df
 original=original[original.driverId.isin(ids)]
 with zipfile.ZipFile(Path(dataset.cache_dir)/'tasks/driver-position.zip') as z:
  archived=pq.read_table(io.BytesIO(z.read(f'driver-position/{split}.parquet')),use_threads=False).to_pandas()
 def canonical(df):return df[['driverId','date','position']].sort_values(['date','driverId']).reset_index(drop=True)
 for comparison in [original,archived]:
  a,b=canonical(reconstructed),canonical(comparison)
  pd.testing.assert_frame_equal(a[['driverId','date']],b[['driverId','date']],check_dtype=False)
  np.testing.assert_allclose(a.position,b.position,rtol=0,atol=1e-12)
 validate_task(reconstructed,ids,dataset.val_timestamp if split=='train' else None)
 past=make_labels(events,times,'past');past=past[past.driverId.isin(ids)]
 known=set(zip(past.driverId,past.date));extra=reconstructed.loc[[k not in known for k in zip(reconstructed.driverId,reconstructed.date)]]
 report[split]=dict(rows=len(reconstructed),timestamps=len(times),nonempty_timestamps=int(reconstructed.date.nunique()),entities=int(reconstructed.driverId.nunique()),released_only_rows=len(extra),first_cutoff=str(times.min()),last_cutoff=str(times.max()),latest_label_end=str(reconstructed.label_end.max()),original_sql='MATCH',archive='MATCH',max_label_error=float(np.max(abs(canonical(reconstructed).position-canonical(archived).position))))
 portable['splits'][split]={'cutoffs':[str(t) for t in times],'db_cutoff':str(dataset.test_timestamp) if split!='test' else None,'expected':json.loads(archived.to_json(orient='records',date_format='iso',double_precision=15))}
 if len(extra):report[split]['example_released_only']=json.loads(extra.head(1).to_json(orient='records',date_format='iso',double_precision=15))[0]
portable['val_timestamp']=str(dataset.val_timestamp)
out=P/'evidence/l152';out.mkdir(parents=True,exist_ok=True)
raw=gzip.compress(json.dumps(portable,separators=(',',':')).encode(),mtime=0);(out/'task-inputs.json.gz').write_bytes(raw)
r=dict(status='PASS',splits=report,total_rows=sum(x['rows'] for x in report.values()),raw_result_rows=len(results),task_payload_sha256=hashlib.sha256(raw).hexdigest(),contract='Complete timestamp grid independently regenerated; identities exact; label absolute tolerance 1e-12; original SQL and released archives compared',cohort_boundary='Past-only still conditions observed labels on future participation; absent future event is not a zero label')
(P/'_task_audit_l152_results.json').write_text(json.dumps(r,indent=2));(P/'_task_sources_l152.json').write_text(json.dumps(dict(commit=commit,sources=manifest,experiment='https://arxiv.org/html/2407.20060v1',blueprint='https://proceedings.mlr.press/v235/fey24a.html'),indent=2));print(json.dumps(r,indent=2))

"""Reconstruct the exact source interval predicates with bounded memory."""
import hashlib,json,shutil,ast
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
P=Path(__file__).resolve().parent;E=P/'evidence/l181';S=P/'sources/l181';packet=E/'packet'
ledger=json.loads((S/'source-ledger.json').read_text());db={}
for name,h in ledger['raw_sha256'].items():
 f=packet/'db'/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==h;db[f.stem]=pd.read_parquet(f)
# Authenticate the original archive against the publication-date source registry.
hashes=json.loads((S/'upstream/relbench/datasets/hashes.json').read_text());print('F1 source registry:',hashes.get('rel-f1/db.zip'))
archive=P/'evidence/l171/rel-f1-db.zip';archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive_hash in json.dumps(hashes.get('rel-f1/db.zip')),'F1 archive differs from pinned v2 registry'
metadata={}
for f in sorted((packet/'db').glob('*.parquet')):
 md=pq.read_metadata(f).metadata
 metadata[f.stem]={k.decode():v.decode() for k,v in md.items() if k.decode() not in ['ARROW:schema','pandas']}
print('METADATA',metadata['results'])
# Source splits use database extrema, not just the target table extrema.
timecols={n:json.loads(m['time_col']) for n,m in metadata.items()}
dbmin=min(db[n][t].min() for n,t in timecols.items() if t)
dbmax=max(db[n][t].max() for n,t in timecols.items() if t)
val=pd.Timestamp('2005-01-01');test=pd.Timestamp('2010-01-01');second=pd.Timedelta(seconds=1)
config=dict(experiment='L181 RelBench v2 F1 autocomplete',revision=ledger['revision'],archive_sha256=archive_hash,db_min=str(dbmin),db_max=str(dbmax),tables=metadata,seeds=list(range(5)),gnn=dict(epochs=10,batch_size=512,channels=128,fanouts=[128,64],lr=.005,loss='L1',selection='validation MAE',clamp_percentiles=[2,98]),budget_usd=10,planned_stop_usd=8,local_cap_seconds=3600,tasks={})
for table,key,expected in [('results','resultId',[8997,1400,4798]),('qualifying','qualifyId',[2228,1854,5733])]:
 task=table+'-position';df=db[table];proxies=['statusId','positionOrder','points','laps','milliseconds','fastestLap','rank'] if table=='results' else []
 taskdir=packet/task;taskdir.mkdir(exist_ok=True)
 counts={};ranges={}
 # pd.date_range descending one second: min == end only for integral-second endpoints.
 assert all(t.value%1000000000==0 for t in [dbmin,dbmax,val,test])
 for split,lo,hi in [('train',dbmin,val-second),('val',val,test-second),('test',test,dbmax)]:
  sub=df[(df.date>lo)&(df.date<=hi)&df.position.notna()]
  out=pd.DataFrame({'entity':sub[key].to_numpy(dtype=np.int64),'time':sub.date.astype('int64').to_numpy(),'y':sub.position.to_numpy(dtype=float)}).sort_values(['time','entity']).reset_index(drop=True)
  assert not out.duplicated(['entity','time']).any();out.to_parquet(taskdir/(split+'.parquet'),index=False);counts[split]=len(out);ranges[split]=[str(lo),str(hi)]
 config['tasks'][task]=dict(table=table,key=key,target='position',proxies=proxies,counts=counts,paper_counts=dict(zip(['train','val','test'],expected)),split_ranges=ranges,counts_match=list(counts.values())==expected,feature_policy='global',baseline_test_fit='train+val')
 print(task,counts,'expected',expected)
config['data_status']='PASS' if all(t['counts_match'] for t in config['tasks'].values()) else 'FAIL_SPLIT_COUNTS'
(packet/'config.json').write_text(json.dumps(config,indent=2)+'\n')
for name in ['examples/gnn_autocomplete.py','examples/model.py','relbench/modeling/nn.py','relbench/modeling/graph.py','relbench/base/task_autocomplete.py','relbench/base/dataset.py','examples/baseline_autocomplete.py']:
 dest=packet/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(S/'upstream'/name,dest)
print('data_status',config['data_status'])

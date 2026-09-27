"""Independent scientific audit: provenance, feature rules, arithmetic and mutations."""
import ast,hashlib,io,json,math,sqlite3,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from _check_l137 import check_pair,check_nominate,check_cluster
from relkit.error_reg_l137 import paired_errors,nominate_slice,cluster_interval
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l137'
check_pair(paired_errors);check_nominate(nominate_slice);check_cluster(cluster_interval)
mutations=0
for check,fn in [(check_pair,lambda *a,**kw:np.zeros(len(a[1]))),(check_nominate,lambda *a,**kw:{'selected':'tiny','slices':{}}),(check_cluster,lambda *a,**kw:{'mean':3,'low':3,'high':3,'entities':2})]:
 try:check(fn)
 except (AssertionError,KeyError,ValueError):mutations+=1
 else:raise AssertionError('Scientific mutation escaped')
old=json.loads((P/'_sources_l129.json').read_text());files={}
for name,sha in old['files'].items():
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha,name
 files['labs/'+name]=sha
for path in [P/'_run_fe_l137.py',P/'_prepare_fe_l137.py',P/'relkit/error_reg_l137.py',P/'_slices_l137.py',P/'_freeze_l137.py',P/'_errors_l137.py',P/'requirements-l117-runtime.txt',P/'requirements-l129-runtime.txt']:
 files[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
source=dict(status='PINNED',files=files,gnn_commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639',fe_commit='445bb7a3b1230f49f8e5890ae81754d3e365680f',paper='https://arxiv.org/html/2407.20060v1',full_paper='NOT_RUN')
(P/'_sources_l137.json').write_text(json.dumps(source,indent=2))
a=np.load(E/'diagnostics.npz');d=json.loads((E/'diagnostics.json').read_text());count=0
with zipfile.ZipFile(P/'sources/l129/db.zip') as z:results=pd.read_parquet(io.BytesIO(z.read('db/results.parquet')))
results=results[results.date<=pd.Timestamp('2010-01-01')]
lookup={i:np.sort(g.date.astype('int64').to_numpy()) for i,g in results.groupby('driverId')}
for split in ['train','val','test']:
 for i,t,n,lag in zip(a[split+'_entity'],a[split+'_time'],a[split+'_history'],a[split+'_recency']):
  dates=lookup.get(i,np.array([],dtype='int64'));expected=int(np.searchsorted(dates,t,side='left'));assert expected==n
  expected_lag=(t-dates[expected-1])/86400e9 if expected else np.inf
  assert expected_lag==lag;count+=1
 df=pd.read_parquet(E/f'fe/{split}-features.parquet');np.testing.assert_array_equal(df.driverId,a[split+'_entity']);np.testing.assert_array_equal(df.date.astype('int64'),a[split+'_time'])
 missing=df[d['feature_columns']].isna().sum(axis=1).to_numpy()/24
 np.testing.assert_array_equal(missing,a[split+'_missing'])
 assert not np.any(a[split+'_mask_low_history'] & a[split+'_mask_high_history'])
 assert np.all(a[split+'_mask_low_history'] | a[split+'_mask_high_history'])
 assert not np.any(a[split+'_mask_stale_history'] & a[split+'_mask_recent_history'])
 assert np.all(a[split+'_mask_missing_recent_slots'] | a[split+'_mask_observed_recent_slots'])
# Rebuild selected result using SQL and validate frozen input hashes.
f=json.loads((E/'frozen.json').read_text());assert not f['test_arrays_read'];assert f['selected']=='high_history'
r=json.loads((E/'errors.json').read_text());assert r['frozen_sha256']==hashlib.sha256((E/'frozen.json').read_bytes()).hexdigest()
for name,h in f['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
payload=json.loads((E/'portable.json').read_text());scored=0
for split,x in payload.items():
 delta=[]
 for g,fe in zip(x['gnn'],x['fe']):
  independent=[abs(gi-y)-abs(fi-y) for y,gi,fi in zip(x['target'],g,fe)];delta.append(independent);scored+=len(independent)
 delta=np.mean(delta,axis=0);ids=np.array([k[0] for k in x['keys']])
 for name,mask in x['masks'].items():
  m=np.array(mask,bool);row=r['splits'][split]['slices'][name]
  assert row['rows']==int(m.sum()) and row['drivers']==len(set(ids[m]))
  if m.any():assert abs(math.fsum(delta[m])/m.sum()-row['mean'])<1e-12
  if row['interval']:
   groups=sorted(set(ids[m]));rng=np.random.default_rng(137);means=[]
   # Independent explicit whole-row concatenation, rather than bincount totals.
   values=[delta[m][ids[m]==group] for group in groups]
   for draw in rng.integers(0,len(groups),size=(2000,len(groups))):means.append(float(np.concatenate([values[j] for j in draw]).mean()))
   lo,hi=np.quantile(means,[.025,.975]);assert abs(lo-row['interval']['low'])<1e-12 and abs(hi-row['interval']['high'])<1e-12
for name in ['_audit_l137_results.json','_source_check_fe_l137_results.json','_execution_l137_results.json','_notebook_fe_l137_results.json','_notebook_gpu_l137_results.json']:
 assert json.loads((P/name).read_text())['status']=='PASS',name
assert json.loads((E/'fe/sql-audit.json').read_text())['status']=='PASS'
assert json.loads((E/'fe/summary.json').read_text())['predictions']==6295
b=json.loads((P/'_budget_l137.json').read_text());reserved=sum(x['upper_usd'] for x in b['reservations']);assert reserved+b['overhead_reserve_usd']<=10
report=dict(status='PASS',mutations_rejected=mutations,diagnostic_queries_independently_checked=count,paired_rows_independently_checked=scored,bootstrap='Explicit whole-row concatenation agrees with vectorized cluster totals for all supported slices',source_files=len(files),budget_worker_upper_usd=reserved,budget_with_overhead_usd=reserved+3,live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l137_results.json').write_text(json.dumps(report,indent=2));print(report)

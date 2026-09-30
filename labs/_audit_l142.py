"""Independent raw-table label reconstruction and archive-to-prediction key audit."""
import hashlib,io,json,zipfile,urllib.request
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent;E=P/'evidence/l142';D=P/'data/l141'
expected={'db.zip':'ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482','driver-position.zip':'775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e'}
D.mkdir(parents=True,exist_ok=True)
for name,digest in expected.items():
 if not (D/name).exists():urllib.request.urlretrieve('https://relbench.stanford.edu/download/rel-f1/'+('tasks/' if name!='db.zip' else '')+name,D/name)
 assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest
with zipfile.ZipFile(D/'db.zip') as z:results=pd.read_parquet(io.BytesIO(z.read('db/results.parquet')));drivers=pd.read_parquet(io.BytesIO(z.read('db/drivers.parquet')))
counts={};comparisons=0;identity_hashes={}
with zipfile.ZipFile(D/'driver-position.zip') as z:
 for split in ['train','val','test']:
  q=pd.read_parquet(io.BytesIO(z.read(f'driver-position/{split}.parquet')));counts[split]=len(q)
  for cutoff,group in q.groupby('date'):
   eligible=results[(results.date>cutoff)&(results.date<=cutoff+pd.Timedelta(days=60))]
   observed=eligible.groupby('driverId').positionOrder.mean().sort_index()
   actual=group.set_index('driverId').position.sort_index()
   assert np.array_equal(observed.index.to_numpy(dtype=np.int64),actual.index.to_numpy(dtype=np.int64)),(split,str(cutoff),'population')
   np.testing.assert_allclose(observed.to_numpy(),actual.to_numpy(),atol=1e-12,rtol=1e-12)
  times=q.date.to_numpy(dtype='datetime64[s]').astype(np.int64);ids=q.driverId.to_numpy(dtype=np.int64)
  keys=list(zip(ids,times));assert len(set(keys))==len(keys)
  identity_hashes[split]=hashlib.sha256(np.array(keys,dtype='<i8').tobytes()).hexdigest()
  if split!='train':
   truth=dict(zip(keys,q.position.to_numpy()))
   for phase in [f'{arm}-{i}' for arm in ['composite','ordinary'] for i in range(5)]:
    a=np.load(E/phase/'predictions.npz');pk=list(zip(a[split+'_entity'],a[split+'_time']))
    assert len(set(pk))==len(pk) and set(pk)==set(keys)
    np.testing.assert_allclose([truth[k] for k in pk],a[split+'_target'],rtol=0,atol=1e-12);comparisons+=len(pk)
report=dict(status='PASS',counts=counts,independently_rebuilt_labels=sum(counts.values()),archive_aligned_predictions=comparisons,query_key_hashes=identity_hashes,archives=expected,
 task_semantics='Average positionOrder in (cutoff, cutoff+60days]; participation in future window determines eligible label rows. Source lower-only one-year eligibility condition is automatically satisfied by a valid future participant.',historical_availability='NOT_ESTABLISHED')
(E/'label-audit.json').write_text(json.dumps(report,indent=2));print(report)

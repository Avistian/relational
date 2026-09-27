"""Independent task-archive census, owner-key checks and MAP cross-checks."""
import io,zipfile,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import torch
from relkit.identity_l132 import mean_average_precision
from relbench.metrics import link_prediction_map
P=Path(__file__).resolve().parent;path=P/'sources/l132/condition-sponsor-run.zip';raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='eeef170e06b6728928116c333f56601e036b24fcbef4f97dc93ec1948700c2f8'
r={};z=zipfile.ZipFile(io.BytesIO(raw));rng=np.random.default_rng(132)
for split,expected in [('train',36934),('val',2081),('test',2057)]:
 name=next(x for x in z.namelist() if x.endswith('/'+split+'.parquet'));df=pd.read_parquet(io.BytesIO(z.read(name)))
 assert len(df)==expected and not df.duplicated(['condition_id','timestamp']).any()
 assert all(len(x)>0 and len(x)==len(set(x)) for x in df.sponsor_id)
 r[split]=dict(queries=len(df),positive_pairs=sum(map(len,df.sponsor_id)),cutoffs=sorted(str(x) for x in df.timestamp.unique()))
 truth=[list(x) for x in df.sponsor_id.iloc[:40]];pred=[]
 for t in truth:
  base=list(dict.fromkeys(t[:4]+rng.choice(10000,20,replace=False).tolist()))[:10];rng.shuffle(base);pred.append(base)
 pred=np.array(pred);isin=np.array([[v in set(t) for v in row] for row,t in zip(pred,truth)]);counts=np.array([len(x) for x in truth])
 official=link_prediction_map(isin,counts);ours=mean_average_precision(pred,truth,10)
 assert abs(official-ours)<1e-12;r[split]['map_check']=dict(official=official,independent=ours)
(P/'_task_audit_l132_results.json').write_text(json.dumps(dict(status='PASS',splits=r,scope='Task archive identities and metrics; label SQL reconstruction NOT_RUN'),indent=2));print(r)

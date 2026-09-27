"""Independently audit all saved pilot query keys, top-k candidates and AP scores."""
from pathlib import Path
import json,numpy as np,pandas as pd
from relkit.identity_l132 import mean_average_precision
P=Path(__file__).resolve().parent;E=P/'evidence/l132';count=0;rows=[]
for variant in ['sage','idgnn']:
 root=E/f'pilot/{variant}-100';r=json.loads((root/'result.json').read_text());done=json.loads((root/'completed.json').read_text());assert done['status']=='COMPLETE'
 a=np.load(root/'predictions.npz')
 for split,n in [('val',2081),('test',2057)]:
  df=pd.read_parquet(E/f'prepared/{split}.parquet');assert len(df)==n and a[split+'_pred'].shape==(n,10)
  assert len(set(zip(a[split+'_source'],a[split+'_time'])))==n
  np.testing.assert_array_equal(a[split+'_source'],df.condition_id);np.testing.assert_array_equal(a[split+'_time'],df.timestamp.astype('int64'))
  pred=a[split+'_pred'];assert (pred>=0).all() and (pred<53241).all()
  score=mean_average_precision(pred,df.sponsor_id.tolist(),10);assert abs(score-r['scores'][split])<1e-12
  rows.append(dict(variant=variant,split=split,queries=n,map=score));count+=n
out=dict(status='PASS',query_rankings=count,candidate_positions=count*10,rows=rows,scope='One-epoch pilots; full five-seed comparison NOT_RUN')
(P/'_audit_pilots_l132_results.json').write_text(json.dumps(out,indent=2));print(out)

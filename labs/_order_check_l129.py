"""Fresh SQL extraction must preserve a declared query order for seeded fitting."""
import json
from pathlib import Path
import pandas as pd
from relkit.fe_experiment_l129 import load_archives,make_features
P=Path(__file__).resolve().parent;S=P/'sources/l129'
tables,labels,_=load_archives(S);sql=(S/'f1/driver-position/feats.sql').read_text()
a=make_features(tables,labels,sql,threads=4);b=make_features(tables,labels,sql,threads=1)
for split in labels:
 for frame in [a[split],b[split]]:
  assert list(zip(frame.driverId,frame.date))==list(zip(labels[split].driverId,labels[split].date)),'SQL ordering is not the frozen archive query order'
 pd.testing.assert_frame_equal(a[split],b[split])
r=dict(status='PASS',query_order='Exact task archive row order',thread_variants=[1,4],feature_values='EXACT across fresh SQL executions')
(P/'_order_check_l129_results.json').write_text(json.dumps(r,indent=2));print(r)

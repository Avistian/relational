"""Compare fresh RDL with explicitly reused, checksum-verified L129 FE predictions."""
import json,hashlib,statistics,math
from pathlib import Path
import numpy as np
from relkit.checkpoint_l130 import keyed_mae
P=Path(__file__).resolve().parent
rdl=json.loads((P/'evidence/l130/summary.json').read_text())
fe=json.loads((P/'evidence/l129/summary.json').read_text())
assert fe['status']=='COMPLETE_RELEASED_PIPELINE_REPLAY'
rows=[];hashes={};count=0
for seed in range(5):
 root=P/f'evidence/l129/paper/seed-{seed}';r=json.loads((root/'result.json').read_text())
 for name,digest in r['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 for name in ['result.json','predictions.npz','model.txt']:
  p=root/name;hashes[str(p.relative_to(P.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
 x=np.load(root/'predictions.npz');y=np.load(P/f'evidence/l130/paper/seed-{seed}/predictions.npz');scores={}
 for split,n in [('val',499),('test',760)]:
  for suffix in ['entity','time','target']:np.testing.assert_array_equal(x[split+'_'+suffix],y[split+'_'+suffix])
  queries=[dict(entity=int(e),time=int(t),target=float(v)) for e,t,v in zip(y[split+'_entity'],y[split+'_time'],y[split+'_target'])]
  preds=[dict(entity=int(e),time=int(t),prediction=float(v)) for e,t,v in zip(x[split+'_entity'],x[split+'_time'],x[split+'_pred'])]
  scores[split]=keyed_mae(queries,list(reversed(preds)))
  assert abs(scores[split]-r['scores'][split])<1e-12;count+=n
 rows.append(dict(seed=seed,**scores))
metrics={}
for split in ['val','test']:
 mean=statistics.mean(r[split] for r in rows);sd=statistics.stdev(r[split] for r in rows)
 assert abs(mean-fe['metrics'][split]['mean'])<1e-12
 metrics[split]=dict(rdl_mean=rdl['metrics'][split]['mean'],rdl_sd=rdl['metrics'][split]['sample_sd'],fe_mean=mean,fe_sd=sd,fe_minus_rdl=mean-rdl['metrics'][split]['mean'])
out=dict(status='PASS',comparison='Fresh L130 basic RDL versus reused L129 tuned manual features',key_and_label_matches=count,metrics=metrics,reused_fe_hashes=hashes,fe_seeds=rows,inference='Descriptive means on one task; seed numbers do not imply paired random draws or equal training budgets',boundaries=fe['protocol_differences'],human_study='NOT_RUN')
(P/'evidence/l130/comparison.json').write_text(json.dumps(out,indent=2));print(metrics)

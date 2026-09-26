"""Independent pooled metric, matched-negatives and selection audit for all six fits."""
import json,hashlib
from pathlib import Path
import numpy as np
from _collect_l103 import metrics
P=Path(__file__).resolve().parent;report=json.loads((P/'_comparison_l103_results.json').read_text());root=P/'results/l103/comparison';error=0.;hashes={}
if not root.exists():root=P/'evidence/l103/comparison'
for seed in range(3):
 a=np.load(root/f'TGAT-{seed}-predictions.npz');b=np.load(root/f'TGN-{seed}-predictions.npz')
 np.testing.assert_array_equal(a['edges'],b['edges']);np.testing.assert_array_equal(a['negative'],b['negative'])
 for arm,z in [('TGAT',a),('TGN',b)]:
  row=next(r for r in report['records'] if r['arm']==arm and r['seed']==seed)
  got=metrics(z['positive'],z['negative_score'])[:2];expected=[row['test']['ap'],row['test']['auc']]
  assert np.max(np.abs(got-expected))<1e-12;error=max(error,float(np.max(np.abs(got-expected))))
  assert row['selected_epoch']==int(np.argmax([x['val']['ap'] for x in row['trace']]))
  assert len(z['edges'])==400
  hashes[f'{arm}-{seed}']=hashlib.file_digest((root/f'{arm}-{seed}-predictions.npz').open('rb'),'sha256').hexdigest()
out={'status':'PASS','fits':6,'matched_negative_and_event_arrays':'EXACT','validation_only_selection':'PASS','independent_metric_max_error':error,'prediction_sha256':hashes}
(P/'_comparison_check_l103_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

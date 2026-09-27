"""Verify the standalone notebook's fresh full run against frozen author outputs."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import nbformat
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);a=parser.parse_args()
report=[];count=0;maximum=0
out=P/'evidence/l129/notebook-full-verification';out.mkdir(exist_ok=True)
for seed in range(5):
 ref=P/f'evidence/l129/paper/seed-{seed}';original=json.loads((ref/'result.json').read_text());new=json.loads((a.directory/f'seed-{seed}.json').read_text())
 assert len(new['trace'])==10 and new['selected_trial']==original['selected_trial']
 for row,old in zip(new['trace'],original['trace']):
  assert row['params']==old['params'] and abs(row['val_mae']-old['val_mae'])<1e-12
 pred=np.load(a.directory/f'seed-{seed}.npz');expected=np.load(ref/'predictions.npz')
 for name in expected.files:
  np.testing.assert_allclose(pred[name],expected[name],rtol=0,atol=1e-10)
  if name.endswith('_pred'):count+=len(pred[name]);maximum=max(maximum,float(np.max(abs(pred[name]-expected[name]))))
 for ext in ['json','npz']:shutil.copyfile(a.directory/f'seed-{seed}.{ext}',out/f'seed-{seed}.{ext}')
 report.append(dict(seed=seed,scores=new['scores'],model_sha256=hashlib.sha256((a.directory/f'seed-{seed}.txt').read_bytes()).hexdigest(),selected_trial=new['selected_trial']))
nb=nbformat.read(P/'solutions/0129-manual-feature-engineering.ipynb',as_version=4);gate=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('RUN_FULL_REPRODUCTION'))
r=dict(status='PASS',full_searches=5,trials=50,selected_refits=5,full_predictions=count,maximum_prediction_difference=maximum,selected_trials_and_parameters='EXACT',fresh_sql_and_preprocessing='EXECUTED in isolated directory',seeds=report,gate_code_sha256=hashlib.sha256(gate.encode()).hexdigest(),scope='Full notebook gate executed as extracted cells in pinned Python3.11 runtime; not live Colab',cloud_usd=0)
(P/'_gate_l129_results.json').write_text(json.dumps(r,indent=2));print(r)

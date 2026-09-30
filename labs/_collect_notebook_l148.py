"""Collect pinned notebook report and independently score the extra history fit."""
from pathlib import Path
import hashlib,json,modal,numpy as np
from relkit.ablation_l148 import keyed_mae
P=Path(__file__).resolve().parent;E=P/'evidence/l148';v=modal.Volume.from_name('l148-ablation-evidence')
for name in ['notebook/execution.json','notebook/cost.json','notebook/l148-report.json','notebook/history-100/result.json','notebook/history-100/predictions.npz']:
 raw=b''.join(v.read_file(name));p=E/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
r=json.loads((E/'notebook/history-100/result.json').read_text());assert r['epochs']==10 and r['arm']=='history' and r['seed']==100
z=np.load(E/'notebook/history-100/predictions.npz');ref=np.load(E/'prepared/queries.npz');count=0
for split in ['val','test']:
 a=list(zip(ref[split+'_entity'],ref[split+'_time']));b=list(zip(z[split+'_entity'],z[split+'_time']));np.testing.assert_array_equal(z[split+'_target'],ref[split+'_target'])
 score=keyed_mae(a,ref[split+'_target'],b,z[split+'_pred']);assert abs(score-r['scores'][split])<1e-10;count+=len(a)
p=P/'results/l148/notebook-executed.ipynb';p.write_bytes(b''.join(v.read_file('notebook/executed.ipynb')))
(E/'notebook/independent-check.json').write_text(json.dumps(dict(status='PASS',extra_predictions=count,excluded_from_primary=True,executed_notebook_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
import nbformat
n=nbformat.read(P/'solutions/0148-ablation-discipline.ipynb',4);code='\n\n'.join(c.source for c in n.cells if c.cell_type=='code')
assert hashlib.sha256(code.encode()).hexdigest()==json.loads((E/'notebook/execution.json').read_text())['original_code_sha256']
budget=json.loads((P/'_budget_l148.json').read_text());costs={str(p.relative_to(E)):json.loads(p.read_text()) for p in E.rglob('*cost.json')}
report=dict(status='PASS',conservative_reservations_plus_overhead=3+sum(r['upper_usd'] for r in budget['reservations']),known_worker_body_usd=sum(c['worker_body_usd'] for c in costs.values()),invoice='NOT_ITEMIZED',failed_preworker_reservations_retained=2,failed_notebook_setup_reservation_retained=True,notebook_predispatch_image_error='no worker dispatched or reservation charged',costs=costs)
(E/'cost-summary.json').write_text(json.dumps(report,indent=2));print(report)

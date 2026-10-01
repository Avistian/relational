"""Reconcile completion and bounded resource accounting after all validations."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l151';b=json.loads((P/'_budget_l151.json').read_text());s=json.loads((E/'summary.json').read_text())
upper=sum(r['upper_usd'] for r in b['reservations'])+b['overhead_reserve_usd'];assert upper<=10
costs={str(p.relative_to(E)):json.loads(p.read_text()) for p in list(E.glob('*-cost.json'))+list(E.glob('notebook/cost.json'))}
r=dict(status='COMPLETE_SELECTED_RELEASED_PROTOCOL',reference=s['tracks']['reference'],selected_course=s['tracks']['selected'],search_lr=s['search']['lr'],primary_fits=10,search_fits=3,epochs_per_fit=20,preprocessing='FRESH',independently_rebuilt_labels=13779,independently_scored_predictions=s['verified_predictions'],query_occurrences=s['query_occurrences'],budget_usd=10,reservations_plus_overhead_usd=upper,worker_body_estimate_usd=sum(c['worker_body_usd'] for c in costs.values()),costs=costs,invoice='NOT_ITEMIZED; worker-body estimate excludes build/startup/commit/storage',historical_identity='NOT_ESTABLISHED',feature_arrival_legality='NOT_ESTABLISHED',whole_paper='NOT_RUN',fresh_fe_comparison='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
for key,file in [('default_notebook','_execution_l151_results.json'),('pinned_notebook','_notebook_l151_results.json'),('delivery','_delivery_l151_results.json')]:r[key]=json.loads((P/file).read_text())['status'] if (P/file).exists() else 'NOT_CHECKED'
(E/'reproduction.json').write_text(json.dumps(r,indent=2));print({k:r[k] for k in ['status','reservations_plus_overhead_usd','worker_body_estimate_usd','default_notebook','pinned_notebook','delivery']})

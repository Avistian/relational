"""Reconcile actual evidence, full portable runs, budgets and delivery checks."""
import hashlib,json,statistics,subprocess,sys
from pathlib import Path
import nbformat
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l155'
read=lambda p:json.loads(p.read_text())
r=read(E/'report.json');b=read(P/'_budget_l155.json');g=read(E/'summary.json');f=read(E/'fe/summary.json')
assert r['scored_prediction_rows']==12590 and r['human_effort_ratio']['status']=='NOT_OBSERVED'
assert r['portfolio']['matched_tasks']==1 and r['figure3_reproduction'].startswith('NOT_RUN')
assert r['test_benefit_driver_bootstrap_95'][0]<0<r['test_benefit_driver_bootstrap_95'][1]
assert all(v==640 for v in g['first_backward_nonfinite'].values())
for name,digest in b['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
reserved=sum(x['upper_usd'] for x in b['reservations']);assert reserved+3<=10 and sum(x['workers'] for x in b['reservations'])<=8
notebook=nbformat.read(P/'solutions/0155-compare-manual-fe.ipynb',4);code='\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
checks={}
for name in ['_check_l155_results.json','_audit_report_l155_results.json','_audit_l155_results.json','_sources_l155.json','_source_check_fe_l155_results.json','_execution_l155_results.json','_delivery_l155_results.json','_notebook_gnn_l155_results.json','_notebook_fe_l155_results.json']:
 report=read(P/name);assert report['status']=='PASS',name;checks[name]=report['status']
 if 'notebook_' in name:assert report['code_sha256']==digest
 if name=='_execution_l155_results.json':assert report['executed_code_sha256']==digest
failed=read(E/'notebook/cost.json');ng=read(P/'_notebook_gnn_l155_results.json');nf=read(P/'_notebook_fe_l155_results.json');initial_fe=read(E/'fe-notebook/first-execution.json')
measured=g['worker_body_usd']+failed['worker_body_usd']+ng['cost']['worker_body_usd']
assert measured<reserved
local_seconds=read(E/'fe/fit-completed.json')['seconds']+nf['seconds']+initial_fe['seconds']+read(E/'fe/pilot/result.json')['seconds']+read(P/'_source_check_fe_l155_results.json')['seconds']+read(E/'fe/preparation.json')['seconds']
assert local_seconds<3600
result=dict(status='PASS',experiment='L155 rel-f1/driver-position — released manual-FE pipeline versus basic RDL',
 primary=dict(gnn_full_fits=5,epochs_per_fit=10,fe_searches=5,trials_per_search=10,prediction_rows=12590,metrics=r['metrics'],test_benefit_driver_bootstrap_95=r['test_benefit_driver_bootstrap_95']),
 primary_gnn_metadata_scope='The reused GNN-only portfolio helper has fresh_fe_comparison=NOT_RUN; the separate FE ledger and combined report establish the L155 comparison. It is not a combined portfolio verdict.',
 notebook=dict(code_sha256=digest,code_cells=48,full_gnn_fits=5,full_fe_searches=5,each_gate_checked_predictions=6295,primary_means_exclude_validation_fits=True),
 budget=dict(hard_cap_usd=10,reserved_worker_resources_usd=reserved,overhead_allowance_usd=3,total_reserved_plus_overhead_usd=reserved+3,worker_body_estimate_usd=measured,invoice='NOT_ITEMIZED',failed_gpu_validation_attempts=1,first_failed_validation_seconds=failed['seconds'],local_recorded_compute_seconds=local_seconds,local_cloud_usd=0),
 audits=checks,source_nonfinite_first_backward_per_seed=g['first_backward_nonfinite'],human_effort='NOT_OBSERVED',human_study='NOT_RUN',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',feature_arrival_legality='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l155_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','budget','human_effort','whole_paper']},indent=2))

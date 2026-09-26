"""Freeze complete verified L116 evidence; retain other lessons' existing records."""
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;R=P.parent
s=json.loads((P/'evidence/l116/summary.json').read_text());assert s['status']=='COMPLETE'
checks={}
for name in ['red','check','trainer_check','source_check','upstream','sparse_input','audit','diagnostics_check','replay','execution','mutation','delivery','visual']:
 path=P/f'_{name}_l116_results.json';r=json.loads(path.read_text());assert r['status']=='PASS',path;checks[path.name]=r['status']
b=json.loads((P/'_budget_l116.json').read_text());assert b['status']=='COMPLETE' and b['maximum_resource_usd']+b['reserved_other_cost_usd']<=10
assert sum(r['workers'] for r in b['reservations'])==13
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sourcepath=P/'_sources_l116.json';sources=json.loads(sourcepath.read_text());assert sha(P/'relkit/debug_l116.py')==sources['source_sha256']
for name,expected in sources['files'].items():assert sha(R/name)==expected
sources['diagnostic_primary_sources']=['https://arxiv.org/abs/1801.07606','https://pytorch-geometric.readthedocs.io/en/2.6.1/tutorial/neighbor_loader.html','https://docs.pytorch.org/docs/stable/notes/autograd.html']
sources['verification_files_sha256']={str(p.relative_to(R)):sha(p) for p in sorted(P.glob('_*l116*.py'))+[P/'requirements-l116-runtime.txt',R/'modal/l116_repro.py']};sourcepath.write_text(json.dumps(sources,indent=2))
paths=[P/'relkit/debug_l116.py',R/'modal/l116_repro.py',P/'l116-reproduction.md',P/'requirements-l116-runtime.txt',R/'lessons/0116-debug-gnn-training.html',R/'lessons/content/0116-debug-gnn-training.md',R/'reference/debug-gnn-training.html',P/'0116-debug-gnn-training.ipynb',P/'solutions/0116-debug-gnn-training.ipynb',P/'html/0116-debug-gnn-training.html',R/'assets/gnn-diagnostics.js',R/'assets/gnn-diagnostics.css',R/'assets/l116-lesson.js']
paths+=sorted((P/'sources/l116').glob('*'));paths+=sorted((P/'evidence/l116').rglob('*'));paths+=sorted((P/'figures/l116').glob('*'));paths+=sorted(P.glob('_*l116*.py'));paths+=sorted(P.glob('_*l116*.json'))
paths=[p for p in paths if p.is_file() and p.name!='_provenance_l116_results.json' and '__pycache__' not in str(p)]
r={'status':'PASS','scope':'Ten fresh complete GCN fits, paired full-data missing-step intervention and local diagnostic probes','checks':checks,'files_sha256':{str(p.relative_to(R)):sha(p) for p in paths},'local_validation_runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__},'inference_node_predictions':1693430,'fresh_fits':10,'fresh_training_epochs':5000,'course_intervention_epochs':160,'pilot_epochs':10,'training_resource_estimate_usd':s['successful_resource_usd'],'historical_identity':'NOT_ESTABLISHED','full_paper_parity':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_provenance_l116_results.json').write_text(json.dumps(r,indent=2))
p=P/'reproductions/execution_evidence.json';all_=json.loads(p.read_text());all_['l116']={'status':'COMPLETE_SELECTED_EXPERIMENT','experiment':'OGB v6 Table6 GCN with repaired instrumented training loop','fresh_runs':10,'epochs':5000,'summary':s['summary'],'independent_original_replay':'1693430 node predictions, zero class mismatches','diagnostic_interventions':'COURSE_ONLY; two 80-epoch full-data runs plus local mechanisms','evidence':'labs/evidence/l116/summary.json','protocol':'labs/l116-reproduction.md','historical_identity':'NOT_ESTABLISHED','whole_paper':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'};p.write_text(json.dumps(all_,indent=2)+'\n');print({'status':'PASS','files_hashed':len(r['files_sha256']),'checks':checks})

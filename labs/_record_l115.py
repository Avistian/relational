"""Freeze verified L115 provenance without modifying other lessons' evidence."""
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;R=P.parent
s=json.loads((P/'evidence/l115/summary.json').read_text());assert s['status']=='COMPLETE'
checks={}
for name in ['check','source_check','upstream','sparse_input','audit','replay','execution','mutation','delivery']:
 path=P/f'_{name}_l115_results.json';r=json.loads(path.read_text());assert r['status']=='PASS',path;checks[path.name]=r['status']
b=json.loads((P/'_budget_l115.json').read_text());assert b['status']=='COMPLETE' and b['maximum_resource_usd']+b['reserved_other_cost_usd']<=10
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sourcepath=P/'_sources_l115.json';sources=json.loads(sourcepath.read_text());assert sha(P/'relkit/patterns_l115.py')==sources['source_sha256']
for name,expected in sources['files'].items():assert sha(R/name)==expected
sources['design_source']='https://arxiv.org/html/1806.01261v3#S4.SS3'
sources['verification_files_sha256']={str(p.relative_to(R)):sha(p) for p in sorted(P.glob('_*l115*.py'))+[P/'requirements-l115-runtime.txt',R/'modal/l115_repro.py']};sourcepath.write_text(json.dumps(sources,indent=2))
paths=[P/'relkit/patterns_l115.py',R/'modal/l115_repro.py',P/'l115-reproduction.md',P/'requirements-l115-runtime.txt',R/'lessons/0115-graph-ml-design-patterns.html',R/'lessons/content/0115-graph-ml-design-patterns.md',R/'reference/graph-ml-design-patterns.html',P/'0115-graph-ml-design-patterns.ipynb',P/'solutions/0115-graph-ml-design-patterns.ipynb',P/'html/0115-graph-ml-design-patterns.html',R/'assets/graph-patterns.js',R/'assets/graph-patterns.css',R/'assets/l115-lesson.js']
paths+=sorted((P/'sources/l115').glob('*'));paths+=sorted((P/'evidence/l115').rglob('*'));paths+=sorted((P/'figures/l115').glob('*'));paths+=sorted(P.glob('_*l115*.py'));paths+=sorted(P.glob('_*l115*.json'))
paths=[p for p in paths if p.is_file() and p.name!='_provenance_l115_results.json' and '__pycache__' not in str(p)]
r={'status':'PASS','scope':'Ten fresh full-data modular GCN fits plus task-readout teaching examples','checks':checks,'files_sha256':{str(p.relative_to(R)):sha(p) for p in paths},'local_validation_runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__},'inference_node_predictions':1693430,'fresh_fits':10,'fresh_training_epochs':5000,'training_resource_estimate_usd':s['successful_resource_usd'],'historical_identity':'NOT_ESTABLISHED','full_paper_parity':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_provenance_l115_results.json').write_text(json.dumps(r,indent=2))
p=P/'reproductions/execution_evidence.json';all_=json.loads(p.read_text());all_['l115']={'status':'COMPLETE_SELECTED_EXPERIMENT','experiment':'OGB v6 Table6 GCN, modular source-preserving implementation','fresh_runs':10,'epochs':5000,'summary':s['summary'],'independent_original_replay':'1693430 node predictions, zero class mismatches','synthetic_examples':'COURSE_ONLY','evidence':'labs/evidence/l115/summary.json','protocol':'labs/l115-reproduction.md','historical_identity':'NOT_ESTABLISHED','whole_paper':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'};p.write_text(json.dumps(all_,indent=2)+'\n');print({'status':'PASS','files_hashed':len(r['files_sha256']),'checks':checks})

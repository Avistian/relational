"""Package verification: measured evidence, artifact identity and honest scope."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent
reports={name:json.loads((P/f'_{name}_l125_results.json').read_text()) for name in ['check','source_check','audit','execution','delivery','teaching']}
assert all(x['status']=='PASS' for x in reports.values())
s=json.loads((P/'evidence/l125/summary.json').read_text());assert s['total_rows']==74063 and len(s['tables'])==9
for name,digest in s['provenance'].items():assert hashlib.sha256((P/'evidence/l125'/name).read_bytes()).hexdigest()==digest,name
with np.load(P/'evidence/l125/encoded-reg.npz',allow_pickle=False) as z:
 assert len(z.files)==18
 for name,record in s['tables'].items():
  ids=z[name+'_ids'];v=z[name+'_vectors'];assert len(ids)==len(set(ids))==record['rows'];assert v.shape==(len(ids),8) and np.isfinite(v).all()
assert reports['audit']['full_data_original_frame_rows']==74063 and reports['audit']['maximum_output_error']==0
assert len(reports['audit']['mutation_checks'])==4 and set(reports['audit']['mutation_checks'].values())=={'REJECTED'}
assert reports['source_check']['model']['output_max_error']==0 and reports['source_check']['model']['gradient_max_error']==0
assert s['gradient']['training_queries']==23 and s['gradient']['visible_results']==18389
assert s['gradient']['max_input_time'][:10]<=s['gradient']['cutoff']
assert all(v>0 for v in s['gradient']['encoder_gradient_l1'].values())
paper=json.loads((P/'_paper_audit_l125_results.json').read_text());assert paper['paper_result']=='NOT_RUN' and len(paper['archive_probes'])==6
budget=json.loads((P/'_budget_l125.json').read_text());assert budget['paid_compute_spend_usd']==0 and not budget['reservations']
from _recover_l125 import inspect_archives
with tempfile.TemporaryDirectory() as tmp:
 assert inspect_archives(tmp)['status']=='BLOCKED_DATA'
 Path(tmp,'db.zip').write_bytes(b'wrong identity');assert inspect_archives(tmp)['files']['db.zip']=='HASH_MISMATCH'
preflight='PASS: missing and mismatching input rejected'
manifest=json.loads((R/'lessons/manifest.json').read_text());row=next(x for x in manifest['lessons'] if x['id']==125)
assert row['slug']=='0125-pytorch-frame-deep-dive'
for file in [R/'lessons/0125-pytorch-frame-deep-dive.html',P/'solutions/0125-pytorch-frame-deep-dive.ipynb',P/'evidence/l125/f1-db.zip',P/'evidence/l125/f1-task.zip',P/'evidence/l125/encoded-reg.npz']:
 assert subprocess.run(['git','check-ignore','-q',str(file)],cwd=R).returncode==1,('Ignored required input',file)
source=json.loads((P/'_sources_l125.json').read_text())
for path,entry in source['files'].items():assert hashlib.sha256((P/path).read_bytes()).hexdigest()==entry['sha256']
files=[P/'relkit/frame_l125.py',P/'_run_l125.py',P/'_check_l125.py',P/'_audit_l125.py',P/'_source_check_l125.py',P/'_build_l125.py',P/'_execute_l125.py',R/'lessons/content/0125-pytorch-frame-deep-dive.md',R/'assets/frame-contract-viz.js']
r={'status':'PASS','feature_rows':74063,'training_query_gradient_check':23,'source_files':len(source['files']),'recovery_preflight':preflight,'checks':{k:v['status'] for k,v in reports.items()},'source_sha256':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'selected_paper_result':'NOT_RUN','historical_identity':'NOT_ESTABLISHED','paid_compute_usd':0,'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner':'PENDING_WRITTEN_DEFENSE','publication':'Not requested; workspace copied-Pages validation, not clean-checkout or live deployment'}
(P/'_verify_l125_results.json').write_text(json.dumps(r,indent=2)+'\n')
ep=P/'reproductions/execution_evidence.json';e=json.loads(ep.read_text());e['lesson_125']={'status':'PREPARED_AND_CHECKED','selected_experiment':'Hu Frame Table2 rel-stackex-engage ResNet+HeteroSAGE ROC-AUC .854','selected_experiment_status':'NOT_RUN','blocking_evidence':'labs/_paper_audit_l125_results.json','course_feature_rows':74063,'real_training_queries':23,'original_frame_max_error':0,'paid_compute_usd':0,'protocol':'labs/l125-reproduction.md','verification':'labs/_verify_l125_results.json','whole_paper':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner':'PENDING_WRITTEN_DEFENSE'};ep.write_text(json.dumps(e,indent=2)+'\n');print(json.dumps(r,indent=2))

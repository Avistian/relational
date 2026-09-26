"""Record verified evidence without changing any other lesson's records."""
import hashlib,json,platform,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
s=json.loads((P/'evidence/l114/summary.json').read_text());assert s['status']=='COMPLETE'
checks={}
for name in ['check','source_check','replay_l114_gcn','replay_l114_mlp','audit','execution','mutation','delivery']:
 filename='_'+name+'_results.json' if name.startswith('replay') else '_'+name+'_l114_results.json'
 report=json.loads((P/filename).read_text());assert report['status']=='PASS',filename;checks[filename]=report['status']
b=json.loads((P/'_budget_l114.json').read_text());assert b['status']=='COMPLETE' and b['maximum_resource_usd']+b['reserved_other_cost_usd']<=10
paths=[P/'relkit/error_l114.py',P/'relkit/ogb_l112.py',R/'modal/l114_repro.py',P/'l114-reproduction.md',P/'requirements-l114-runtime.txt',R/'lessons/0114-ogb-error-analysis.html',R/'lessons/content/0114-ogb-error-analysis.md',R/'reference/ogb-error-analysis.html',P/'0114-ogb-error-analysis.ipynb',P/'solutions/0114-ogb-error-analysis.ipynb',R/'assets/error-slices.js',R/'assets/error-slices.css',R/'assets/l114-lesson.js']
paths+=sorted((P/'sources/l114').glob('*'));paths+=sorted((P/'evidence/l114').rglob('*'));paths+=sorted((P/'figures/l114').glob('*'));paths+=sorted(P.glob('_*l114*.py'));paths+=sorted(P.glob('_*l114*.json'))
paths=[p for p in paths if p.is_file() and p.name!='_provenance_l114_results.json' and '__pycache__' not in str(p)]
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
r={'status':'PASS','scope':'Complete selected MLP experiment plus full retrospective census and fresh replay of inherited GCN','checks':checks,'files_sha256':hashes,'local_validation_runtime':{'python':sys.version,'platform':platform.platform()},'inference_node_predictions':3386860,'fresh_mlp_fits':10,'fresh_mlp_epochs':5000,'fresh_gcn_training':'NOT_RUN_IN_L114; ten L112 states reused','training_resource_estimate_usd':s['successful_resource_usd'],'historical_identity':'NOT_ESTABLISHED','full_paper_parity':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(P/'_provenance_l114_results.json').write_text(json.dumps(r,indent=2))
p=P/'reproductions/execution_evidence.json';all_=json.loads(p.read_text());all_['l114']={'status':'COMPLETE_SELECTED_EXPERIMENT','experiment':'OGB v6 Table6 MLP plus course slice census','fresh_runs':10,'epochs':5000,'gcn_training':'REUSED_L112','summary':s['summary'],'validation_nominated_slice':{'family':s['selected_validation_slice']['family'],'slice':s['selected_validation_slice']['slice'],'test_n':s['same_rule_on_test']['n'],'test_delta_pp':s['same_rule_on_test']['mean_delta_pp']},'original_prediction_replay':'EXACT_ALL_TWENTY_STATES','independent_node_predictions':3386860,'budget':b,'historical_identity':'NOT_ESTABLISHED','full_paper_reproduction':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','protocol':'labs/l114-reproduction.md','evidence':'labs/evidence/l114/summary.json','delivery':'PASS'};p.write_text(json.dumps(all_,indent=2)+'\n')
print({'status':'PASS','checks':len(checks),'hashed_files':len(hashes),'scope':'L114 only'})

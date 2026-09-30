"""Final scientific/delivery ledger; exact identities and evidence boundaries."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
import nbformat
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l141'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=load(E/'training.json');a=load(E/'label-audit.json');d=load(E/'diagnosis.json');source=load(P/'_sources_l141.json')
for name,item in source['files'].items():assert sha(P/'sources/l141'/name)==item['sha256']
for name,digest in s['source_artifact_hashes'].items():assert sha(E/name)==digest
ids=[];rows=[];query_occurrences=0
for seed in range(5):
 r=load(E/f'seed-{seed}/result.json');ident=load(E/f'seed-{seed}-started.json');ids.append(ident['uuid'])
 assert ident['trainer_sha256']==sha(P/'_full_l141.py') and ident['model_sha256']==sha(P/'relkit/relgnn_l141.py')
 assert r['kind']=='RECONSTRUCTED_TRAINING' and r['seed']==seed and r['epochs']==10
 assert r['best_epoch']==1+int(np.argmin([x['val_mae'] for x in r['history']]))
 assert all(x['queries']==7453 and x['steps']==15 for x in r['history'])
 assert r['temporal_audit']['future_violations']==0;query_occurrences+=r['temporal_audit']['query_occurrences'];rows.append(r)
assert len(set(ids))==5
for split in ['val','test']:
 x=np.array([r['scores'][split]['mae'] for r in rows]);assert abs(x.mean()-s['reconstruction'][split]['mean'])<1e-12;assert abs(x.std(ddof=1)-s['reconstruction'][split]['sample_sd'])<1e-12
assert s['reconstruction']['test']['descriptive_closeness']=='OUTSIDE_TOLERANCE'
assert a['independently_rebuilt_labels']==8712 and a['archive_aligned_predictions']==7554
assert sum(d['matched_nonfinite_gradients'].values())==640 and d['max_finite_gradient_error']<1e-6
replay=load(E/'replay-compatible/result.json');assert replay['kind']=='CHECKPOINT_COMPATIBILITY_REPLAY'
assert replay['compatibility']['before']['position']=='categorical' and replay['compatibility']['after']['position']=='numerical'
assert (E/'replay-failure.txt').is_file()
reports={name:load(P/f'_{name}_l141_results.json') for name in ['parity','execution','notebook','delivery']}
assert all(r['status']=='PASS' for r in reports.values())
assert reports['execution']['executed_code_sha256']==reports['notebook']['code_sha256']==reports['delivery']['notebook_code_sha256']
assert reports['notebook']['additional_predictions_independently_scored']==2518
budget=load(P/'_budget_l141.json');reserved=sum(x['upper_usd'] for x in budget['reservations'])+budget['overhead_reserve_usd'];assert reserved<=10
costs=[]
for file in E.rglob('*cost.json'):
 x=load(file)
 if 'worker_body_usd' in x:costs.append({'artifact':str(file.relative_to(E)),'usd':x['worker_body_usd']})
# Successful diagnosis timing is separately recorded; unsuccessful attempts are conservatively reserved.
measured=sum(x['usd'] for x in costs)+d['seconds']*.00022572
budget['accounting']=dict(conservative_reserved_plus_overhead_usd=reserved,recorded_worker_body_usd=measured,recorded_worker_body_scope='Known body times only; excludes unrecorded failed diagnostics/startup/build/commit/storage; invoice NOT_ITEMIZED',cost_artifacts=costs)
(P/'_budget_l141.json').write_text(json.dumps(budget,indent=2))
html=(R/'lessons/0141-composite-message-passing.html').read_text()
for token in ['OUTSIDE_TOLERANCE','CHECKPOINT_COMPATIBILITY_REPLAY','640','8,712','PENDING_WRITTEN_DEFENSE','4.259311']:assert token in html,token
report=dict(status='PASS',primary_fresh_fits=5,full_epochs_per_fit=10,checkpoint_compatible_replays=1,primary_predictions=7554,independent_labels=8712,additional_notebook_predictions=2518,
 primary_training_query_occurrences=query_occurrences,source_hashes='PASS',all_primary_source_checkpoints='NUMERIC_CLOSE',gradient_health='MATCHED_NONFINITE_ENTRIES_DISCLOSED',
 notebook_full_gate='ONE_EXTRA_FULL_FIT_AND_COMPATIBLE_REPLAY',conservative_usd=reserved,recorded_worker_body_usd=measured,
 score_status='OUTSIDE_TOLERANCE',historical_training='NOT_ESTABLISHED',whole_paper='NOT_RUN',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l141_results.json').write_text(json.dumps(report,indent=2));print(report)
ledger=P/'reproductions/execution_evidence.json';j=load(ledger);j['lesson_141']=dict(status='PREPARED_AND_CHECKED',experiment='RelGNN rel-f1/driver-position Table2',replay=s['replay'],reconstruction=s['reconstruction'],protocol='labs/l141-reproduction.md',verification='labs/_verify_l141_results.json',**{k:report[k] for k in ['historical_training','whole_paper','learner','conservative_usd']});ledger.write_text(json.dumps(j,indent=2)+'\n')

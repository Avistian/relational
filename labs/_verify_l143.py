"""Independent completion/provenance accounting; does not assume a score outcome."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from relkit.reproduction_l143 import first_validation_min,evidence_verdict,verify_numeric_layout
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l143'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=load(E/'training.json');a=load(E/'label-audit.json');d=load(E/'diagnosis.json');source=load(P/'_sources_l143.json')
for name,item in source['files'].items():assert sha(P/'sources/l141'/name)==item['sha256']
for name,digest in s['source_artifact_hashes'].items():assert sha(E/name)==digest
ids=[];rows=[];queries=0
for seed in range(5):
 r=load(E/f'seed-{seed}/result.json');ident=load(E/f'seed-{seed}-started.json');ids.append(ident['uuid'])
 assert ident['trainer_sha256']==sha(P/'_full_l143.py') and ident['model_sha256']==sha(P/'relkit/relgnn_l143.py')
 assert r['kind']=='RECONSTRUCTED_TRAINING' and r['seed']==seed and r['epochs']==10
 assert r['best_epoch']==first_validation_min(r['history'])
 assert all(x['queries']==7453 and x['steps']==15 for x in r['history'])
 assert r['temporal_audit']['future_violations']==0;queries+=r['temporal_audit']['query_occurrences'];rows.append(r)
 assert all(v['max_original_error']<=2e-4 for v in r['scores'].values())
 assert sum(r['nonfinite_gradients'].values())>0
assert len(set(ids))==5
records=[dict(seed=r['seed'],kind=r['kind'],epochs=r['epochs'],complete=r['status']=='COMPLETE',test_mae=r['scores']['test']['mae']) for r in rows]
verdict=evidence_verdict(records);assert verdict==s['verdict'];assert verdict['score']==s['reconstruction']['test']['descriptive_closeness']
for split in ['val','test']:
 x=np.array([r['scores'][split]['mae'] for r in rows]);assert abs(x.mean()-s['reconstruction'][split]['mean'])<1e-12;assert abs(x.std(ddof=1)-s['reconstruction'][split]['sample_sd'])<1e-12
assert a['independently_rebuilt_labels']==8712 and a['archive_aligned_predictions']==7554
assert sum(d['matched_nonfinite_gradients'].values())==640 and d['max_finite_gradient_error']<1e-6
replay=load(E/'replay-compatible/result.json');assert replay['kind']=='CHECKPOINT_COMPATIBILITY_REPLAY'
e=replay['compatibility']['evidence'];moments={c:(e['mean']['table'][i],e['std']['table'][i]) for i,c in enumerate(['number','position'])}
assert verify_numeric_layout(['number','position'],moments,e['mean']['checkpoint'],e['std']['checkpoint'])
assert (E/'checkpoint-incompatibility.txt').is_file()
prepared=load(E/'prepared/prepared.json');compatible=load(E/'compatible/prepared.json')
assert compatible['parent_graph_sha256']==prepared['graph_sha256'] and replay['graph_sha256']==compatible['graph_sha256']
assert all(r['graph_sha256']==prepared['graph_sha256'] for r in rows)
reports={name:load(P/f'_{name}_l143_results.json') for name in ['check','parity','execution','notebook']}
assert all(r['status']=='PASS' for r in reports.values())
assert reports['execution']['executed_code_sha256']==reports['notebook']['code_sha256']
assert reports['notebook']['additional_predictions_independently_scored']==2518
budget=load(P/'_budget_l143.json');reserved=sum(x['upper_usd'] for x in budget['reservations'])+budget['overhead_reserve_usd'];assert reserved<=10
measured=sum(load(f)['worker_body_usd'] for f in E.rglob('*cost.json'))+d['seconds']*.00022572
budget['accounting']=dict(conservative_reserved_plus_overhead_usd=reserved,recorded_worker_body_usd=measured,scope='Body estimates exclude startup/build/commit/storage; overhead reserved; invoice NOT_ITEMIZED')
(P/'_budget_l143.json').write_text(json.dumps(budget,indent=2))
report=dict(status='PASS',primary_fresh_fits=5,full_epochs_per_fit=10,checkpoint_compatible_replays=1,primary_predictions=7554,independent_labels=8712,additional_notebook_predictions=2518,primary_training_query_occurrences=queries,source_hashes='PASS',all_primary_source_predictions='NUMERIC_CLOSE',gradient_health='MATCHED_NONFINITE_ENTRIES_DISCLOSED',notebook_full_gate='ONE_EXTRA_FULL_FIT_AND_COMPATIBLE_REPLAY',conservative_usd=reserved,recorded_worker_body_usd=measured,score_status=verdict['score'],historical_training='NOT_ESTABLISHED',whole_paper='NOT_RUN',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l143_results.json').write_text(json.dumps(report,indent=2));print(report)
ledger=P/'reproductions/execution_evidence.json';j=load(ledger);j['lesson_143']=dict(status='PREPARED_AND_CHECKED',experiment='RelGNN rel-f1/driver-position Table2',replay=s['replay'],reconstruction=s['reconstruction'],protocol='labs/l143-reproduction.md',verification='labs/_verify_l143_results.json',**{k:report[k] for k in ['historical_training','whole_paper','learner','conservative_usd']});ledger.write_text(json.dumps(j,indent=2)+'\n')

"""Independent provenance, completion, score and budget checks for all four tracks."""
import hashlib,json
from pathlib import Path
import nbformat,numpy as np
from relkit.reproduction_l143 import first_validation_min,verify_numeric_layout
from relkit.checkpoint_l150 import select_candidate,summarize_track,checkpoint_verdict
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l150'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read(E/'summary.json');a=read(E/'label-audit.json');d=read(E/'diagnosis.json');budget=read(P/'_budget_l150.json');prepared=read(E/'prepared/prepared.json')
for name,item in read(E/'sources.json')['files'].items():assert sha(P/'sources/l141'/name)==item['sha256']
for name,digest in s['files'].items():assert sha(E/name)==digest
for reservation in budget['reservations']:
 for name,digest in reservation['sources'].items():assert sha(P/name)==digest
assert select_candidate(s['search']['candidates'])==s['search']['lr']==.003
for name,digest in s['search']['files'].items():assert sha(R/name)==digest
ids=[];queries=0
for record in s['records']:
 phase=record['phase'];r=read(E/phase/'result.json');ident=read(E/(phase+'-started.json'));ids.append(ident['uuid'])
 assert ident['trainer_sha256']==sha(P/'_full_l150.py') and ident['model_sha256']==sha(P/'relkit/relgnn_l143.py')
 assert r['status']=='COMPLETE' and r['track']==record['track']
 assert r['temporal_audit']['future_violations']==0;queries+=r['temporal_audit']['query_occurrences']
 assert all(v['max_original_error']<2e-4 for v in r['scores'].values())
 if record['track']!='replay':
  assert r['kind']=='RECONSTRUCTED_TRAINING' and r['epochs']==10 and len(r['history'])==10
  assert r['best_epoch']==first_validation_min(r['history']) and r['selection_mae']==min(x['val_mae'] for x in r['history'])
  assert all(x['queries']==7453 and x['steps']==15 for x in r['history'])
  assert r['graph_sha256']==prepared['graph_sha256']
  assert sha(P/'results/l150'/f'{phase}.pt')==r['checkpoint_sha256']
 if record['track']=='search':
  assert set(r['scores'])=={'val'} and r['test_access']=='FORBIDDEN'
  assert not any(k.startswith('test') for k in np.load(E/phase/'predictions.npz').files)
 assert record['nonfinite_entries']==sum(r['nonfinite_gradients'].values())
assert len(set(ids))==14
for track,seeds in [('reference',list(range(5))),('selected',list(range(10,15)))]:
 rows=[r for r in s['records'] if r['track']==track];agg=summarize_track(rows,track,seeds)
 assert agg['mean']==s['tracks'][track]['mean'] and agg['sample_sd']==s['tracks'][track]['sample_sd']
 assert checkpoint_verdict(True,agg['mean'],False,False)==s['tracks'][track]['gates']
assert a['independently_rebuilt_labels']==8712 and a['archive_aligned_predictions']==s['predictions']==15346
assert sum(d['matched_nonfinite_gradients'].values())==640 and d['max_finite_gradient_error']<1e-6
replay=read(E/'replay-compatible/result.json');assert replay['kind']=='CHECKPOINT_COMPATIBILITY_REPLAY'
e=replay['compatibility']['evidence'];moments={c:(e['mean']['table'][i],e['std']['table'][i]) for i,c in enumerate(['number','position'])};assert verify_numeric_layout(['number','position'],moments,e['mean']['checkpoint'],e['std']['checkpoint'])
compatible=read(E/'compatible/prepared.json');assert compatible['parent_graph_sha256']==prepared['graph_sha256'] and replay['graph_sha256']==compatible['graph_sha256']
assert (E/'checkpoint-incompatibility.txt').is_file()
book=nbformat.read(P/'solutions/0150-q3-reproduction-checkpoint.ipynb',4);code='\n\n'.join(c.source for c in book.cells if c.cell_type=='code');digest=hashlib.sha256(code.encode()).hexdigest()
default=read(P/'_execution_l150_results.json');notebook=read(P/'_notebook_l150_results.json')
assert default['status']==notebook['status']=='PASS'
assert default['executed_code_sha256']==notebook['code_sha256']==digest
assert notebook['additional_predictions_independently_scored']==2518 and notebook['full_training']=='VALIDATION_ONE'
reserved=sum(r['upper_usd'] for r in budget['reservations'])+budget['overhead_reserve_usd'];assert abs(reserved-9.636168)<1e-10 and reserved<=10
cost=sum(read(p)['worker_body_usd'] for p in E.rglob('*cost.json'))+d['seconds']*budget['rate']
budget['accounting']=dict(reserved_plus_overhead_usd=reserved,recorded_worker_body_estimate_usd=cost,invoice='NOT_ITEMIZED',scope='Body estimates exclude startup/build/commit/storage; overhead reserved')
(P/'_budget_l150.json').write_text(json.dumps(budget,indent=2)+'\n')
report=dict(status='PASS',primary_fits=10,search_fits=3,epochs_per_fit=10,primary_and_search_predictions=15346,independent_labels=8712,additional_notebook_predictions=2518,query_occurrences_audited=queries,source_hashes='PASS',selection='VALIDATION_ONLY_HASH_FROZEN',selected_lr=.003,reference=s['tracks']['reference'],selected=s['tracks']['selected'],gradient_health='MATCHED_NONFINITE_ENTRIES_DISCLOSED',notebook_code_sha256=digest,notebook_full_gate='ONE_EXTRA_FULL_FIT_AND_REPLAY',reserved_plus_overhead_usd=reserved,recorded_worker_body_estimate_usd=cost,invoice='NOT_ITEMIZED',historical_identity='NOT_ESTABLISHED',competitive='NOT_ESTABLISHED',whole_paper='NOT_RUN',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l150_results.json').write_text(json.dumps(report,indent=2));print(report)
p=P/'reproductions/execution_evidence.json';ledger=read(p);ledger['lesson_150']=dict(status='PREPARED_AND_CHECKED',experiment='RelGNN F1 reference/search/selected checkpoint',verification='labs/_verify_l150_results.json',protocol='labs/l150-reproduction.md',tracks=s['tracks'],selected_lr=.003,historical_identity='NOT_ESTABLISHED',competitive='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE');p.write_text(json.dumps(ledger,indent=2)+'\n')

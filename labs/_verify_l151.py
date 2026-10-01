"""Independent source, identity, metric, selection and completeness audit."""
import ast,hashlib,json,statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from relkit.trial_l139 import keyed_auc,trial_target
from relkit.portfolio_l151 import select_candidate,summarize_track,portfolio_verdict
from _check_l151 import check_select,check_summary,check_entry
P=Path(__file__).resolve().parent;E=P/'evidence/l151';S=P/'sources/l151'
check_select(select_candidate);check_summary(summarize_track);check_entry(portfolio_verdict)
manifest=json.loads((S/'manifest.json').read_text())
for name,row in manifest['files'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==row['sha256']
def nodes(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=nodes(P/'relkit/rdl_l117.py');same=[]
for name,file in [('Model','examples__model.py'),('HeteroEncoder','relbench__modeling__nn.py'),('HeteroTemporalEncoder','relbench__modeling__nn.py'),('HeteroGraphSAGE','relbench__modeling__nn.py'),('make_pkey_fkey_graph','relbench__modeling__graph.py'),('get_node_train_table_input','relbench__modeling__graph.py')]:
 original=nodes(S/file)[name]
 if name=='Model':original.body=[n for n in original.body if not(isinstance(n,ast.FunctionDef) and n.name=='forward_dst_readout')]
 assert ast.dump(original,include_attributes=False)==ast.dump(canonical[name],include_attributes=False);same.append(name)
audit=json.loads((E/'prepared/task_audit.json').read_text());assert audit['status']=='PASS' and audit['label_mismatches']==audit['eligibility_mismatches']==0
truth=np.load(E/'prepared/queries.npz');counts={s:len(truth[s+'_target']) for s in ['train','val','test']};assert counts=={'train':11994,'val':960,'test':825}
for rows in json.loads((E/'prepared/samples.json').read_text()).values():
 for r in rows:assert trial_target(r['start'],r['analyses'],0)==(True,r['target'])
records=[];histories={};portable={};verified=0;occurrences=0;uuids=[];all_hashes={};gradient_counts={}
for phase in [f'ref-{s}' for s in range(5)]+[f'search-{s:03d}' for s in [50,100,200]]+[f'selected-{s}' for s in range(10,15)]:
 root=E/phase;r=json.loads((root/'result.json').read_text());z=np.load(root/'predictions.npz');identity=json.loads((E/(phase+'-started.json')).read_text());uuids.append(identity['uuid'])
 assert identity['trainer_sha256']==hashlib.sha256((P/'_full_l151.py').read_bytes()).hexdigest()
 assert identity['model_sha256']==hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest()
 assert r['status']=='COMPLETE' and r['epochs']==len(r['history'])==20
 assert r['best_epoch']==1+int(np.argmax([h['val']['roc_auc'] for h in r['history']]))
 assert r['selection_auc']==max(h['val']['roc_auc'] for h in r['history'])
 assert all(h['queries']==11994 and h['steps']==24 for h in r['history'])
 assert r['original_model_parity']['status']=='NUMERIC_CLOSE'
 assert r['temporal_audit']['future_violations']==0
 splits=['val'] if r['track']=='search' else ['val','test']
 assert set(r['scores'])==set(splits)
 if r['track']=='search':assert r['test_access']=='FORBIDDEN' and not any(k.startswith('test') for k in z.files)
 assert r['temporal_audit']['query_occurrences']==20*(11994+960)+960+(825 if 'test' in splits else 0)
 occurrences+=r['temporal_audit']['query_occurrences'];histories[phase]=r['history']
 row=dict(phase=phase,track=r['track'],seed=r['seed'],lr=r['lr'],epochs=r['epochs'],complete=True,selection_auc=r['selection_auc'])
 for split in splits:
  for field in ['study','time','target']:assert np.array_equal(z[split+'_'+field],truth[split+'_'+field])
  keys=list(zip(z[split+'_study'],z[split+'_time']));order=np.random.default_rng(151).permutation(len(keys))
  value=keyed_auc(keys,z[split+'_target'],[keys[i] for i in order],z[split+'_pred'][order]);assert abs(value-roc_auc_score(z[split+'_target'],z[split+'_pred']))<1e-12
  assert abs(value-r['scores'][split]['roc_auc'])<1e-12;verified+=len(keys);row[split+'_auc']=value
 for key in z.files:portable[phase+'_'+key]=z[key]
 trace=np.load(root/'sampled_trace.npz')
 for key in trace.files:
  if key.endswith('_times'):assert (trace[key]<=trace['cutoffs'][trace[key[:-6]+'_owners']]).all()
 assert hashlib.sha256((P/'results/l151'/f'{phase}.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']
 gradient_counts[phase]=sum(r['first_batch_nonfinite_gradients'].values())
 all_hashes[phase]=dict(checkpoint=r['checkpoint_sha256'],predictions=hashlib.sha256((root/'predictions.npz').read_bytes()).hexdigest(),trainer=identity['trainer_sha256']);records.append(row)
assert len(set(uuids))==13
frozen=json.loads((E/'frozen.json').read_text());assert select_candidate(frozen['candidates'])==frozen['lr']
assert not frozen['test_access'] and frozen['selected_seeds']==list(range(10,15))
for path,h in frozen['files'].items():assert hashlib.sha256((P.parent/path).read_bytes()).hexdigest()==h
for r in records:
 if r['track']=='selected':assert r['lr']==frozen['lr']
tracks={}
for track,seeds in [('reference',list(range(5))),('selected',list(range(10,15)))]:
 rows=[r for r in records if r['track']==track];value=summarize_track(rows,track,seeds);value.update(val_mean=statistics.mean(r['val_auc'] for r in rows),val_sample_sd=statistics.stdev(r['val_auc'] for r in rows),verdict=portfolio_verdict(track,True,value['mean'],False));tracks[track]=value
mutants=[]
for label,check,fn in [('choose_smallest_auc',check_select,lambda rows:min(rows,key=lambda r:r['selection_auc'])['lr']),('ignore_completeness',check_summary,lambda rows,t,seeds:dict(mean=.68,sample_sd=np.sqrt(.00025))),('score_implies_mastery',check_entry,lambda *args:dict(paper_score='CLOSE',learner='DEFENDED'))]:
 try:check(fn)
 except (ValueError,AssertionError,KeyError):mutants.append(label)
 else:raise AssertionError('Mutant survived')
gradient=json.loads((E/'gradient-audit.json').read_text());assert gradient['status']=='PASS'
summary=dict(records=records,tracks=tracks,search=frozen,histories=histories,verified_predictions=verified,query_occurrences=occurrences,first_batch_nonfinite_counts=gradient_counts,gradient_audit=gradient,hashes=all_hashes)
(E/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(E/'portable.npz',**portable)
b=json.loads((P/'_budget_l151.json').read_text());upper=sum(r['upper_usd'] for r in b['reservations'])+b['overhead_reserve_usd'];assert upper<=10
report=dict(status='PASS',source_ast_exact=same,counts=counts,independent_labels=sum(counts.values()),verified_predictions=verified,unique_runs=len(set(uuids)),query_occurrences=occurrences,first_batch_nonfinite_counts=gradient_counts,gradient_audit=gradient,rejected_mutants=mutants,reservations_plus_overhead_usd=upper,historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_l151_results.json').write_text(json.dumps(report,indent=2));print(report)

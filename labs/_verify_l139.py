"""Independent source, real-query, temporal and prediction identity checks."""
import ast,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from relkit.trial_l139 import trial_target,visibility_mask,keyed_auc,rank_auc
from _check_l139 import check_target,check_visibility,check_score
P=Path(__file__).resolve().parent;E=P/'evidence/l139';S=P/'sources/l139'
for check,fn in [(check_target,trial_target),(check_visibility,visibility_mask),(check_score,keyed_auc)]:check(fn)
manifest=json.loads((S/'manifest.json').read_text())
for name,row in manifest['files'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==row['sha256'],name
def nodes(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=nodes(P/'relkit/rdl_l117.py');same=[]
for name,file in [('Model','examples__model.py'),('HeteroEncoder','relbench__modeling__nn.py'),('HeteroTemporalEncoder','relbench__modeling__nn.py'),('HeteroGraphSAGE','relbench__modeling__nn.py'),('make_pkey_fkey_graph','relbench__modeling__graph.py'),('get_node_train_table_input','relbench__modeling__graph.py')]:
 original=nodes(S/file)[name]
 if name=='Model':original.body=[n for n in original.body if not(isinstance(n,ast.FunctionDef) and n.name=='forward_dst_readout')]
 assert ast.dump(original,include_attributes=False)==ast.dump(canonical[name],include_attributes=False),name;same.append(name)
identity=json.loads((E/'data_identity.json').read_text());assert all(v['match'] and v['sha256']==v['historical'] for v in identity.values())
report=json.loads((E/'task_audit.json').read_text());assert report['label_mismatches']==report['eligibility_mismatches']==0
assert [report['splits'][s]['rows'] for s in ['train','val','test']]==[11994,960,825]
samples=json.loads((E/'samples.json').read_text());examples=0
for rows in samples.values():
 for row in rows:assert trial_target(row['start'],row['analyses'],0)==(True,row['target']);examples+=1
# Guard the actual false-negative found in the independent checker.
a=pd.DataFrame({'timestamp':[0,0],'nct_id':pd.Series([1,2],dtype='Int64')});b=a.astype({'nct_id':'int64'})
assert not a.set_index(['timestamp','nct_id']).index.equals(b.set_index(['timestamp','nct_id']).index)
a['nct_id']=a.nct_id.astype('int64');assert a.set_index(['timestamp','nct_id']).index.equals(b.set_index(['timestamp','nct_id']).index)
mutants=[]
for label,check,fn in [('missing_as_negative',check_target,lambda start,analyses,t: (True,0)),('include_cutoff',check_target,lambda start,analyses,t:trial_target(start,analyses,t-1e-9)),('batch_max',check_visibility,lambda times,owners,cutoffs:np.asarray(times)<=max(cutoffs)),('ignore_prediction_keys',check_score,lambda q,y,p,s:rank_auc(y,s)),('tie_as_win',check_score,lambda q,y,p,s:1.)]:
 try:check(fn)
 except (AssertionError,ValueError):mutants.append(label)
 else:raise AssertionError('mutant survived:'+label)
summary=json.loads((E/'training.json').read_text());queries=np.load(E/'queries.npz');count=0;scored={}
for seed in range(5):
 result=json.loads((E/f'full/seed-{seed}/result.json').read_text());z=np.load(E/f'full/seed-{seed}/predictions.npz')
 assert result['epochs']==20 and len(result['history'])==20
 assert result['best_epoch']==int(np.argmax([r['val']['roc_auc'] for r in result['history']]))+1
 assert result['original_model_parity']['status']=='NUMERIC_CLOSE'
 expected_occurrences=20*(11994+960)+960+825
 assert result['temporal_audit']['query_occurrences']==expected_occurrences and result['temporal_audit']['future_violations']==0
 for split in ['val','test']:
  for field in ['study','time','target']:assert np.array_equal(z[split+'_'+field],queries[split+'_'+field])
  keys=list(zip(z[split+'_study'],z[split+'_time']));order=np.random.default_rng(seed).permutation(len(keys))
  value=keyed_auc(keys,z[split+'_target'],[keys[i] for i in order],z[split+'_pred'][order])
  assert abs(value-roc_auc_score(z[split+'_target'],z[split+'_pred']))<1e-12
  assert abs(value-result['scores'][split]['roc_auc'])<1e-12
  count+=len(keys);scored[f'{seed}-{split}']=value
r=dict(status='PASS',source_ast_exact=same,archive_hashes='MATCH',real_examples=examples,task_queries=13779,verified_predictions=count,rejected_mutants=mutants,original_selected_model_parity='NUMERIC_CLOSE on512real roots per seed; rtol1e-5 atol1e-6',query_occurrences=summary['audited_query_occurrences'],scores=scored,historical_identity='NOT_ESTABLISHED')
(P/'_verify_l139_results.json').write_text(json.dumps(r,indent=2));print(r)

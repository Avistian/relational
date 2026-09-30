"""Independent source, schedule, query, arithmetic and evidence-gate checks."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from relkit.checkpoint_l140 import select_checkpoint,align_predictions,reproduction_verdict
from relkit.amazon_l138 import rank_auc
from _check_l140 import check_selection,check_alignment,check_verdict
P=Path(__file__).resolve().parent;E=P/'evidence/l140'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nodes(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for check,fn in [(check_selection,select_checkpoint),(check_alignment,align_predictions),(check_verdict,reproduction_verdict)]:check(fn)
canonical=nodes(P/'relkit/rdl_l117.py');same=[]
for name,file in [('Model','examples__model.py'),('HeteroEncoder','relbench__modeling__nn.py'),('HeteroTemporalEncoder','relbench__modeling__nn.py'),('HeteroGraphSAGE','relbench__modeling__nn.py'),('make_pkey_fkey_graph','relbench__modeling__graph.py'),('get_node_train_table_input','relbench__modeling__graph.py')]:
 original=nodes(P/'sources/l139'/file)[name]
 if name=='Model':original.body=[n for n in original.body if not(isinstance(n,ast.FunctionDef) and n.name=='forward_dst_readout')]
 assert ast.dump(original,include_attributes=False)==ast.dump(canonical[name],include_attributes=False),name;same.append(name)
inherited=json.loads((E/'inherited_provenance.json').read_text())
for name,digest in inherited['files'].items():assert sha(P.parent/name)==digest,name
alignment=json.loads((P/'evidence/l138/paper_alignment.json').read_text())
assert alignment['splits']['train']['row_difference']==-24172
assert all(alignment['splits'][split]['row_difference']==0 for split in ['val','test'])
trial_audit=json.loads((P/'evidence/l139/task_audit.json').read_text())
assert [trial_audit['splits'][split]['rows'] for split in ['train','val','test']]==[11994,960,825]
source_files={}
for n in [138,139]:
 manifest=json.loads((P/f'sources/l{n}/manifest.json').read_text())
 for name,row in manifest['files'].items():
  path=P/f'sources/l{n}'/name;assert sha(path)==row['sha256'];source_files[str(path.relative_to(P))]=row
# Executed loop, loss and audit behavior must match prior released trainer.
new=nodes(P/'_full_l140.py')['full_run'];old=nodes(P/'_full_l139.py')['full_run']
for name in ['audit_batch','predict']:
 a=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
a=next(n for n in new.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
b=next(n for n in old.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
mutants=[]
for name,check,fn in [('last_tie',check_selection,lambda h:max(h,key=lambda r:(r['val']['roc_auc'],r['epoch']))['epoch']),('test_selection',check_selection,lambda h:max(h,key=lambda r:r['test'])['epoch']),('positional_zip',check_alignment,lambda q,k,p:np.asarray(p)),('partial_average',check_verdict,lambda v,s,t,e,p:dict(execution='COMPLETE',mean=float(np.mean(list(v.values()))),sample_sd=float(np.std(list(v.values()),ddof=1)) if len(v)>1 else 0.,score='CLOSE',protocol='GAPPED',historical_identity='NOT_ESTABLISHED')),('score_implies_history',check_verdict,lambda *a:dict(reproduction_verdict(*a),historical_identity='MATCH'))]:
 try:check(fn)
 except (AssertionError,ValueError,KeyError):mutants.append(name)
 else:raise AssertionError('Faulty implementation survived: '+name)
# Agreement with a separate pairwise AUROC implementation on tied fixtures.
rng=np.random.default_rng(140)
for _ in range(50):
 y=rng.integers(0,2,30);s=rng.integers(0,5,30)/4
 pair=(s[y==1,None]>s[y==0]).mean()+.5*(s[y==1,None]==s[y==0]).mean()
 assert abs(rank_auc(y,s)-pair)<1e-12
summary=json.loads((E/'training.json').read_text());preflight=json.loads((E/'preflight.json').read_text());count=0;cost=0.;uuids=set()
for file,digest in summary['source_artifact_hashes'].items():assert sha(E/file)==digest,file
for task,n,entity,epochs in [('amazon',138,'customer',10),('trial',139,'study',20)]:
 base=np.load(P/(f'evidence/l{n}/recency_predictions.npz' if task=='amazon' else f'evidence/l{n}/queries.npz'))
 values={'val':{},'test':{}}
 for seed in range(5):
  folder=E/task/f'seed-{seed}';r=json.loads((folder/'result.json').read_text());z=np.load(folder/'predictions.npz');ident=json.loads((folder/'run_identity.json').read_text())
  assert ident['run_uuid'] not in uuids;uuids.add(ident['run_uuid']);assert ident['source_sha256']==sha(P/'_full_l140.py')
  assert r['epochs']==epochs and r['best_epoch']==select_checkpoint(r['history'])
  cfg=r['config'];assert cfg['aggr']==('sum' if task=='amazon' else 'mean') and cfg['lr']==(.005 if task=='amazon' else .0001) and cfg['fanout']==([128,64] if task=='amazon' else [64,32])
  expected=epochs*((2001*512 if task=='amazon' else 11994)+len(base['val_target']))+len(base['val_target'])+len(base['test_target'])
  assert r['temporal_audit']['query_occurrences']==expected and r['temporal_audit']['future_violations']==0
  assert r['original_model_parity']['status']=='NUMERIC_CLOSE' and r['original_model_parity']['real_query_roots']==512
  for split in ['val','test']:
   keys=list(zip(base[split+'_'+entity],base[split+'_time']));pk=list(zip(z[split+'_entity'],z[split+'_time']));assert np.array_equal(z[split+'_target'],base[split+'_target'])
   pred=align_predictions(keys,pk,z[split+'_pred']);metric=roc_auc_score(base[split+'_target'],pred);assert abs(metric-r['scores'][split]['roc_auc'])<1e-12;count+=len(pred);values[split][seed]=metric
  cost+=json.loads((E/f'{task}-seed-{seed}-cost.json').read_text())['resource_usd']
 for split in ['val','test']:
  expected=summary['tasks'][task]['metrics'][split];actual=reproduction_verdict(values[split],list(range(5)),expected['paper_target'],.01,task!='amazon')
  for key in ['mean','sample_sd']:assert abs(actual[key]-expected[key])<1e-12,(task,split,key)
  for key in ['execution','score','protocol']:assert actual[key]==expected[key],(task,split,key)
 for name,digest in preflight[task]['archives'].items():assert digest==cfg['archives'][name]
cost=sum(json.loads(p.read_text())['resource_usd'] for p in E.glob('*-cost.json'))
budget=json.loads((P/'_budget_l140.json').read_text());ceiling=sum(r['upper_usd'] for r in budget['reservations'])+budget['overhead_reserve_usd'];assert ceiling<=10
manifest=dict(commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639',files=source_files,adaptations=['Task configuration parameterized; original epoch loop and audit/predict functions AST-identical to L139.','Archive and graph hashes checked before every fit; preprocessing reused.','Task output identity field normalized to entity; run UUIDs and source hashes added.','Unused recommendation forward_dst_readout omitted; classification path identical.'],full_trainer_sha256=sha(P/'_full_l140.py'),model_sha256=sha(P/'relkit/rdl_l117.py'),runtime_sha256=sha(P/'requirements-l117-runtime.txt'),inherited_label_audits={str(n):sha(P/f'evidence/l{n}/task_audit.json') for n in [138,139]})
(P/'sources/l140/manifest.json').write_text(json.dumps(manifest,indent=2))
report=dict(status='PASS',source_ast_exact=same,trainer_epoch_loop='AST_EXACT',mutants_rejected=mutants,tied_auc_fixtures=50,verified_predictions=count,unique_fresh_runs=len(uuids),resource_body_estimate_usd=cost,reservation_ceiling_with_overhead_usd=ceiling,invoice='NOT_ITEMIZED',historical_identity='NOT_ESTABLISHED',label_reconstruction='INHERITED_L138_L139; checksum-linked, not freshly rerun',real_ingestion_history='UNAVAILABLE')
(P/'_verify_l140_results.json').write_text(json.dumps(report,indent=2));print(report)

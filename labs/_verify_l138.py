"""Independent source, query and score checks for L138."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from relkit.amazon_l138 import review_target,rank_auc,visible_paths
from _check_l138 import check_target,check_paths,check_auc
P=Path(__file__).resolve().parent;E=P/'evidence/l138';S=P/'sources/l138'
for check,fn in [(check_target,review_target),(check_paths,visible_paths),(check_auc,rank_auc)]:check(fn)
m=json.loads((S/'manifest.json').read_text())
for name,row in m['files'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==row['sha256']
probe=json.loads((E/'probe.json').read_text());assert all(x['historical_match'] for x in probe['archives'].values())
def nodes(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
canonical=nodes(P/'relkit/rdl_l117.py');same=[]
for name,file in [('Model','examples__model.py'),('HeteroEncoder','relbench__modeling__nn.py'),('HeteroTemporalEncoder','relbench__modeling__nn.py'),('HeteroGraphSAGE','relbench__modeling__nn.py'),('make_pkey_fkey_graph','relbench__modeling__graph.py'),('get_node_train_table_input','relbench__modeling__graph.py')]:
 original=nodes(S/file)[name]
 if name=='Model':original.body=[n for n in original.body if not (isinstance(n,ast.FunctionDef) and n.name=='forward_dst_readout')]
 assert ast.dump(canonical[name],include_attributes=False)==ast.dump(original,include_attributes=False),name
 same.append(name)
samples=json.loads((E/'samples.json').read_text());sample_count=0
for split,rows in samples.items():
 for row in rows:assert review_target(row['events'],0)==(True,row['target']);sample_count+=1
z=np.load(E/'recency_predictions.npz');audit=json.loads((E/'task_audit.json').read_text());scores={}
for split in ['val','test']:
 y=z[split+'_target'];s=z[split+'_score'];ids=z[split+'_customer'];t=z[split+'_time'];counts=z[split+'_count']
 assert len(y)==audit['splits'][split]['rows'] and len(set(zip(ids,t)))==len(y)
 assert (counts>0).all() and (s>=0).all() and (s<91).all()
 got=rank_auc(y,s);assert abs(got-roc_auc_score(y,s))<1e-12;assert abs(got-audit['splits'][split]['recency_auc'])<1e-12
 assert abs(average_precision_score(y,s)-audit['splits'][split]['recency_ap'])<1e-12
 scores[split]=dict(queries=len(y),auc=got)
# Mutants must fail load-bearing checks.
mutants=[]
for label,check,fn in [('left_inclusive',check_target,lambda days,t,horizon=91:(bool(np.any((np.array(days)>=t-horizon)&(np.array(days)<=t))),1)),('ties_as_wins',check_auc,lambda y,s:float(np.mean([a>=b for a in np.array(s)[np.array(y)==1] for b in np.array(s)[np.array(y)==0]]))),('ignore_cutoff',check_paths,lambda edges,times,root,cutoff,hops=2:visible_paths(edges,{k:None for k in times},root,cutoff,hops))]:
 try:check(fn)
 except (AssertionError,ValueError):mutants.append(label)
 else:raise AssertionError('mutant survived:'+label)
r=dict(status='PASS',source_ast_exact=same,archive_hashes='historical MATCH',full_audited_queries=sum(x['rows'] for x in audit['splits'].values()),label_mismatches=0,real_example_reconstruction=sample_count,heldout_scores=scores,rejected_mutants=mutants,full_training='See separate reproduction ledger',live_colab='NOT_CHECKED')
(P/'_verify_l138_results.json').write_text(json.dumps(r,indent=2));print(r)

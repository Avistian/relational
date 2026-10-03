"""Independent course audit, evidence corruption tests and inherited model parity."""
import ast,copy,hashlib,json,tempfile,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from _audit_b07 import audit_course
from _source_b07 import audit_sources
from _test_b07 import check_intervention,check_selection,check_pairing
from relkit.semantic_b07 import intervene,fit_probe,paired_delta
from relkit.carte_b07 import load_encoder,make_graph,batch_graphs,normalize_numeric
P=Path(__file__).resolve().parent;E=P/'evidence/b07'
check_intervention(intervene);check_selection(fit_probe);check_pairing(paired_delta)
r=audit_course(P,E/'runs');(E/'course-audit.json').write_text(json.dumps(r,indent=2)+'\n')
s=audit_sources(P/'sources/b07');(E/'source-gate.json').write_text(json.dumps(s,indent=2)+'\n')
original=json.loads((E/'runs/results.json').read_text());rejected=[]
for case in ['missing_run','duplicate_run','row_order','label','source_id','prediction','selected_alpha','protocol','schema']:
 c=copy.deepcopy(original)
 if case=='missing_run':c['records'].pop()
 elif case=='duplicate_run':c['records'][-1]=c['records'][0]
 elif case=='row_order':c['records'][0]['test_ids'].reverse()
 elif case=='label':c['records'][0]['target'][0]+=1
 elif case=='source_id':c['records'][0]['source_ids'][0]+=1
 elif case=='prediction':c['records'][0]['head']['prediction'][0]+=1
 elif case=='selected_alpha':c['records'][0]['head']['alpha']=999
 elif case=='protocol':c['protocol_sha256']='bad'
 elif case=='schema':c['records'][0]['columns'][0]='target'
 try:audit_course(P,E/'runs',c)
 except (ValueError,AssertionError):rejected.append(case)
 else:raise AssertionError('Corruption accepted: '+case)
wrong=[]
for check,fn in [(check_intervention,lambda f,a:f.copy()),(check_selection,lambda *a,**k:dict(alpha=100,mean=[1],scale=[1],coefficient=[0],validation_mse=[],prediction=[0,0])),(check_pairing,lambda l,r:[a-b for a,b in zip(l.values(),r.values())])]:
 try:check(fn)
 except (ValueError,AssertionError,KeyError):wrong.append(check.__name__)
 else:raise AssertionError('Wrong learner function passed')
# Exact AST equality to the inherited visible architecture, not a new architecture claim.
a=ast.parse((P/'relkit/carte_b07.py').read_text());b=ast.parse((P/'relkit/carte_l074.py').read_text());old={n.name:ast.dump(n) for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for n in a.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert old[n.name]==ast.dump(n),n.name
state=torch.load(P/'data/l074/kg_pretrained.pt',weights_only=True,map_location='cpu');sel=torch.load(P/'data/b07/selected_checkpoint.pt',weights_only=True,map_location='cpu')
assert all(torch.equal(v,state[k]) for k,v in sel.items())
v=np.load(P/'data/b07/vectors.npz');vectors=dict(zip(v['names'].tolist(),v['vectors']));model=load_encoder(P/'data/b07/selected_checkpoint.pt',True,0).eval()
graphs=[make_graph({'ABV':float(i),'name':'red'},vectors) for i in (1,2,3)]
x,index,edge,heads=batch_graphs(graphs)
with torch.no_grad():np.testing.assert_allclose(model(x,index,edge)[heads],model.rows(x,index,edge,heads),rtol=1e-5,atol=1e-6)
f=pd.read_parquet(P/'data/b07/wine_pl.parquet');tr=np.arange(64);n1,policy=normalize_numeric(f,tr);g=f.copy();g.loc[64:,'ABV']=1e8;n2,_=normalize_numeric(g,tr);np.testing.assert_allclose(n1.iloc[tr].select_dtypes(include='number'),n2.iloc[tr].select_dtypes(include='number'),equal_nan=True)
run=subprocess.run([sys.executable,str(P/'_reproduce_b07.py'),'--run'],capture_output=True,text=True);assert run.returncode!=0 and 'INCOMPLETE_SOURCE_PROTOCOL' in run.stderr
out=dict(status='PASS',fits=r['fits'],test_predictions=r['test_predictions'],evidence_corruptions_rejected=rejected,wrong_learner_functions_rejected=wrong,inherited_architecture_ast='EXACT',selected_checkpoint_tensors='EXACT',full_graph_vs_center_only='PASS',heldout_numeric_perturbation='PASS',paper_dispatch='REFUSED',learner='PENDING_WRITTEN_DEFENSE')
(P/'_verify_b07_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
print('Aggregates:',json.dumps(r['aggregate']));print('Paired:',json.dumps(r['paired']))

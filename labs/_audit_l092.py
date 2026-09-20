"""Independent protocol, saved-prediction, upstream-probe and provenance audit."""
import ast,contextlib,hashlib,io,json,re
from pathlib import Path
import numpy as np
import torch
import nbformat
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent / "relkit"))
from han_l092 import load_acm,knn_probe,HAN,masked_loss
P=Path(__file__).resolve().parent;m=json.loads((P/'_sources_l092.json').read_text());torch.set_num_threads(1)
for name,digest in m['files'].items():assert hashlib.sha256((P/'sources/han-l092'/name).read_bytes()).hexdigest()==digest
x,y,edges,train,val,test,meta=load_acm(P/'data/l092',m)
for edge in edges:
 key=edge[0]*len(x)+edge[1];assert bool(torch.all(key[1:]>key[:-1]));assert sum((edge[0]==edge[1]).tolist())==len(x)
# Execute the original released probe body on synthetic data, without rewriting its internals.
source=(P/'sources/han-l092/jhyexp.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='my_KNN')
namespace={'np':np,'KNeighborsClassifier':KNeighborsClassifier,'f1_score':f1_score}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned original my_KNN>','exec'),namespace)
rng=np.random.RandomState(5);features=rng.normal(size=(100,6));labels=np.arange(100)%3
np.random.seed(123);capture=io.StringIO()
with contextlib.redirect_stdout(capture):namespace['my_KNN'](features,labels)
ours=knn_probe(features,labels,np.arange(100),123)
for line,fraction in zip(capture.getvalue().splitlines(),[.2,.4,.6,.8]):
 match=re.search(r'f1_macro: ([0-9.]+), f1_micro: ([0-9.]+)',line);assert match
 for metric,value in zip(['macro_f1','micro_f1'],match.groups()):
  assert abs(np.mean([r[metric] for r in ours if r['fraction']==fraction])-float(value))<=.000051
# Permutation equivariance and paper-global remote-node intervention on a small graph.
torch.manual_seed(11);model=HAN(4).double().eval();xx=torch.randn(5,4,dtype=torch.float64)
e=torch.stack([torch.arange(5),torch.arange(5)])
with torch.no_grad():
 z,_,beta=model(xx,[e,e]);perm=torch.tensor([4,1,0,3,2]);zp,_,bp=model(xx[perm],[e,e])
 torch.testing.assert_close(zp,z[perm]);torch.testing.assert_close(bp,beta[perm])
 remote=xx.clone();remote[-1]*=4;zz,_,_=model(remote,[e,e]);torch.testing.assert_close(zz[0],z[0])
 model.mode='paper_global';a,_,_=model(xx,[e,e]);b,_,_=model(remote,[e,e]);assert not torch.allclose(a[0],b[0])
report=json.loads((P/'_paper_l092_results.json').read_text())
assert report['data']==meta and len(report['runs'])==1
for run in report['runs']:
 assert run['epoch_ceiling']==200 and len(run['knn'])==40
 best_a=0.;best_l=float('inf');selected=None;waiting=0
 for t in run['trace']:
  va,vl=t['validation_accuracy'],t['validation_ce']
  if va>=best_a or vl<=best_l:
   if va>=best_a and vl<=best_l:selected=t['epoch']
   best_a=max(best_a,va);best_l=min(best_l,vl);waiting=0
  else:waiting+=1
 assert selected==run['selected_epoch']
 assert len(run['trace'])==200 or waiting==100
 assert run['test_ids']==test.tolist() and run['test_labels']==y[test].tolist()
 assert abs(np.mean(np.array(run['test_predictions'])==y[test].numpy())-run['test_accuracy'])<1e-7
 for r in run['knn']:
  tr,ts=set(r['train_ids']),set(r['test_ids']);assert not tr&ts and tr|ts==set(test.tolist())
  assert len(tr)==int(len(test)*r['fraction'])
  yy=y[r['test_ids']].numpy()
  for metric,average in [('macro_f1','macro'),('micro_f1','micro')]:assert abs(f1_score(yy,r['predictions'],average=average)-r[metric])<1e-12
 for row in report['table3_acm']:
  records=[r for r in run['knn'] if r['fraction']==row['fraction']]
  for metric,prefix in [('macro_f1','macro'),('micro_f1','micro')]:
   values=[r[metric] for r in records];assert abs(np.mean(values)-row[prefix+'_mean'])<1e-12;assert abs(np.std(values,ddof=1)-row[prefix+'_split_sd'])<1e-12
 student=nbformat.read(P/'0092-meta-paths.ipynb',as_version=4)
 assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==3
 assert not any('from relkit' in c.source for c in student.cells if c.cell_type=='code')
result={'status':'PASS','source_hashes':len(m['files']),'full_size_sorted_unique_supports':'PASS','original_release_knn_body_vs_port':'PASS','node_permutation_equivariance':'PASS','fixed_weight_remote_intervention':'PASS','checkpoint_rule_reconstruction':'PASS','all_saved_probe_predictions_and_f1':'PASS','student_live_tasks':3,'historical_tensorflow_execution':'NOT_RUN','full_paper_parity':'NOT_ESTABLISHED'}
(P/'_audit_l092_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

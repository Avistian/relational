"""Mechanism mutations and complete tiny trainer smoke; no paper score claim."""
import json,pickle,tempfile
from pathlib import Path
import numpy as np
import torch
from relkit.cvitkovic_l118 import *
from _check_l118 import check_extract,check_normalize,check_pool
P=Path(__file__).resolve().parent
torch.set_num_threads(1)
check_extract(rdb_to_graph);check_normalize(normalized_sum);check_pool(attention_pool)
mutants=[('undirected_closure',check_extract,lambda n,e,t:(undirected_hops(n,e,t,n),e)),('plain_mean',check_normalize,lambda x,e:x.mean(0).expand_as(x)),('batch_softmax',check_pool,lambda v,g,b,n:torch.stack([(v*torch.softmax(g,0)[:,None])[b==i].sum(0) for i in range(n)]))]
rejected=[]
for name,check,fn in mutants:
 try:check(fn)
 except AssertionError:rejected.append(name)
 else:raise AssertionError('Surviving mutant '+name)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);data=root/'graphs';data.mkdir()
 info={'train_dp_ids':list(range(100)),'node_type_to_int':{'A':0},'label_feature':'A.TARGET','node_types_and_features':{'A':{'x':{'type':'SCALAR','RobustScaler_center_':0,'RobustScaler_scale_':1},'TARGET':{'type':'CATEGORICAL','sorted_values':[0,1]}}}}
 for i in range(100):
  with (data/str(i)).open('wb') as f:pickle.dump(([],[0],[],{'A':{'x':[float(i%2)*2-1]}},i%2),f)
 result=fit_fold(info,data,0,root/'run',epochs=3,batch_size=8,hidden=8,dropout=0)
 pred=np.load(root/'run/predictions.npz');assert len(pred['ids'])==20
 expected=list(released_folds(info['train_dp_ids']))[0][2];assert np.array_equal(pred['ids'],expected)
 assert abs(roc_auc_score(pred['y'],pred['p'])-result['test_auroc'])<1e-12
 try:fit_fold(info,data,0,root/'timed-out',deadline=time.monotonic()-1,epochs=3,batch_size=8,hidden=8)
 except TimeoutError:assert json.loads((root/'timed-out/incomplete.json').read_text())['status']=='INCOMPLETE'
 else:raise AssertionError('Deadline not enforced')
# Source epsilon/clipping/missing and unknown categorical values.
assert torch.allclose(scalar_encode([14,None,30],10,2),torch.tensor([[2.,0.],[0.,1.],[5.,0.]]),atol=2e-7)
s={'status':'PASS','mutations_rejected':rejected,'trainer_smoke':'3 epochs, 100 synthetic graphs; complete save/reload/evaluation path','deadline':'PASS','full_data_training':'NOT_RUN'}
(P/'_verify_l118_results.json').write_text(json.dumps(s,indent=2));print(s)

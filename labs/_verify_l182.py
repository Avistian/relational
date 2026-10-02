"""Independent scalar, adversarial and source-operator checks for L182."""
import hashlib,importlib.util,itertools,json,math,sys
from pathlib import Path
import numpy as np
from relkit.composite_l182 import legal_fusion,attend,keyed_auc,factorial_interaction
from _check_l182 import checks
P=Path(__file__).resolve().parent
assert checks(legal_fusion,keyed_auc,factorial_interaction)=='PASS'
rng=np.random.default_rng(182);cases=0;error=0.
for n in [0,1,2,9]:
 for _ in range(12):
  q=rng.normal(size=(3,2));m=rng.normal(size=(n,2));d=rng.integers(0,3,n)
  actual,w=attend(q,m,d);expected=np.zeros_like(q)
  for j in range(3):
   rows=[i for i in range(n) if d[i]==j]
   logits=[sum(float(q[j,k])*float(m[i,k]) for k in range(2))/math.sqrt(2) for i in rows]
   exps=[math.exp(v-max(logits)) for v in logits]
   for i,v in zip(rows,exps):
    for k in range(2):expected[j,k]+=v/sum(exps)*m[i,k]
  np.testing.assert_allclose(actual,expected,atol=1e-12);error=max(error,float(np.max(abs(actual-expected))));cases+=1
wrong=[(lambda *a:(np.zeros((2,2)),np.array([0,1])),keyed_auc,factorial_interaction),(legal_fusion,lambda *a:.5,factorial_interaction),(legal_fusion,keyed_auc,lambda x:x[:,3]-x[:,0])]
for fs in wrong:
 try:checks(*fs)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Incorrect learner solution accepted')
# Authenticate upstream before differential checking the exact one-head specialization.
ledger=json.loads((P/'_sources_l143.json').read_text());path=P/'sources/l141/examples__relgnn_conv.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()==ledger['files'][path.name]['sha256']
import torch
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('original182',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
model=module.RelGNNConv('dim-fact-dim',2,2,1,'sum',bias=False).double().eval()
with torch.no_grad():
 for parameter in model.parameters():parameter.zero_()
 for layer in [model.aggr_conv.lin_l,model.aggr_conv.lin_r,model.lin_query,model.lin_key,model.lin_value,model.final_proj]:layer.weight.copy_(torch.eye(2,dtype=torch.float64))
source=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64,requires_grad=True)
bridge=torch.tensor([[0.,1.],[2.,0.]],dtype=torch.float64,requires_grad=True)
query=torch.tensor([[1.,0.]],dtype=torch.float64,requires_grad=True)
ea=torch.tensor([[0,1],[0,0]]);eg=torch.tensor([[0,1],[0,1]])
out,fused=model((source,bridge,query),(ea,eg));expected,_=attend(query.detach().numpy(),(source+bridge).detach().numpy(),np.array([0,0]))
np.testing.assert_allclose(out.detach().numpy(),expected,atol=1e-12)
# Finite difference on inputs checks the nontrivial attention derivative at this state.
out.sum().backward();grad_error=0.
for tensor,which in [(source,0),(bridge,1),(query,2)]:
 for index in itertools.product(range(tensor.shape[0]),range(2)):
  arrays=[source.detach().numpy().copy(),bridge.detach().numpy().copy(),query.detach().numpy().copy()]
  arrays[which][index]+=1e-6;plus=attend(arrays[2],arrays[0]+arrays[1],np.array([0,0]))[0].sum()
  arrays[which][index]-=2e-6;minus=attend(arrays[2],arrays[0]+arrays[1],np.array([0,0]))[0].sum()
  err=abs((plus-minus)/2e-6-float(tensor.grad[index]));grad_error=max(grad_error,err);assert err<1e-8
r=dict(status='PASS',scalar_attention_cases=cases,max_scalar_error=error,incorrect_learner_functions_rejected=3,original_relgnn_one_head_output='PASS',finite_difference_max_gradient_error=grad_error,scope='Identity projections; destination skip zero; one head; not full RelGNN training parity',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'_verify_l182_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

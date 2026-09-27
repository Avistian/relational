"""Independent PyG parity, update, row permutation and empty/absent semantics."""
import copy,json,warnings
from pathlib import Path
import torch
from torch_geometric.nn import HeteroConv,SAGEConv
from relkit.hetero_l133 import explicit_layer,audit_layer,tiny_experiment
from _check_l133 import check_neighbors,check_relation,check_merge
from relkit.hetero_l133 import sum_neighbors,relation_output,merge_relations
torch.set_num_threads(1);torch.manual_seed(133)
e={('a','r','b'):torch.tensor([[0,1,1],[0,0,2]]),('b','s','b'):torch.empty((2,0),dtype=torch.long),('b','rev','a'):torch.tensor([[0,2],[0,1]])}
x={'a':torch.randn(2,3),'b':torch.randn(4,3)}
c=HeteroConv({k:SAGEConv((3,3),3,aggr='sum') for k in e},aggr='sum')
state={k:v.clone() for k,v in c.state_dict().items()};rng=torch.get_rng_state().clone()
r=audit_layer(c,x,e)
assert torch.equal(rng,torch.get_rng_state())
for k,v in state.items():torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)
assert all(p.grad is None for p in c.parameters())
assert r.get('audit_dtype')=='float64' and r.get('audit_device')=='cpu'
a=copy.deepcopy(c);b=copy.deepcopy(c)
ref=a(x,e);out,_=explicit_layer(b,x,e)
for m,y in [(a,ref),(b,out)]:
 opt=torch.optim.Adam(m.parameters(),lr=.005);sum(z.square().mean() for z in y.values()).backward();opt.step()
for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)
# Different permutations for each table. Transform BOTH source and destination indices.
p={'a':torch.tensor([1,0]),'b':torch.tensor([2,0,3,1])};inv={k:torch.argsort(v) for k,v in p.items()}
ep={k:torch.stack([inv[k[0]][v[0]],inv[k[2]][v[1]]]) for k,v in e.items()}
y,_=explicit_layer(c,{k:x[k][v] for k,v in p.items()},ep);base,terms=explicit_layer(c,x,e)
for k in y:torch.testing.assert_close(y[k],base[k][p[k]])
absent={k:v for k,v in e.items() if k!=('b','s','b')};without,_=explicit_layer(c,x,absent)
torch.testing.assert_close(base['b']-without['b'],terms[('b','s','b')],atol=1e-6,rtol=1e-5)
# Edge order changes must preserve the multiset, including duplicates.
rev={k:v.flip(1) for k,v in e.items()};reordered,_=explicit_layer(c,x,rev)
for k in base:torch.testing.assert_close(base[k],reordered[k])
# Prove learner checks reject three semantic mistakes.
mutants=[]
for name,check,fn in [('lost_duplicate',check_neighbors,lambda x,e,n:sum_neighbors(x,torch.unique(e,dim=1),n)),('no_root',check_relation,lambda c,x,d,e:c.lin_l(sum_neighbors(x,e,len(d)))),('relation_mean',check_merge,lambda o:{k:v/2 for k,v in merge_relations(o).items()})]:
 try:check(fn)
 except (AssertionError,RuntimeError,ValueError):mutants.append(name)
 else:raise AssertionError('Survived mutant '+name)
with warnings.catch_warnings():
 warnings.simplefilter('ignore',UserWarning);fit=tiny_experiment()
r.update(status='PASS',adam_update='MATCH',independent_table_permutation='MATCH',edge_order='MATCH',empty_vs_absent='DIFFERENT_AS_EXPECTED',mutants_rejected=mutants,tiny_fit=fit,torch=torch.__version__)
Path('labs/_verify_l133_results.json').write_text(json.dumps(r,indent=2));print(r)

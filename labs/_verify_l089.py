"""Behavioral checks independent of the trainer: edges, release arithmetic, gradient and leakage."""
import ast,hashlib,json,types
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from cluster_gcn_l089 import *
LAB=Path(__file__).resolve().parent
torch.set_num_threads(1)
a=sp.csr_matrix(np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]],dtype=np.float32))
parts=[np.array([0,1]),np.array([2,3])]
ids,b=induced_batch(a,parts,[1,0]);np.testing.assert_array_equal(ids,[2,3,0,1]);assert b.nnz==6
_,single=induced_batch(a,parts,[0]);assert single.nnz==2
s=enhanced_support(a).toarray()
np.testing.assert_allclose(s[1],[1/3,2/3,1/3,0],rtol=1e-6)
# Execute unmodified normalization functions from the archived release (no TF runtime required).
source=(LAB/'sources/l089/utils.py').read_text();nodes=ast.parse(source).body
ns={'np':np,'sp':sp}
for node in nodes:
 if isinstance(node,ast.FunctionDef) and node.name in ['normalize_adj','normalize_adj_diag_enhance']:
  exec(compile(ast.Module(body=[node],type_ignores=[]),'release/utils.py','exec'),ns)
np.testing.assert_allclose(s,ns['normalize_adj_diag_enhance'](a,1).toarray(),rtol=1e-6)
np.testing.assert_allclose(enhanced_support(a,-1).toarray(),ns['normalize_adj'](a).toarray())
# Source-level q=1 partition oracle exposes binary support vs doubled raw loops.
partition_source=(LAB/'sources/l089/partition_utils.py').read_text()
partition_node=next(n for n in ast.parse(partition_source).body if isinstance(n,ast.FunctionDef) and n.name=='partition_graph')
ns2={'sp':sp,'time':time,'tf':types.SimpleNamespace(logging=types.SimpleNamespace(info=lambda *args:None))}
exec(compile(ast.Module(body=[partition_node],type_ignores=[]),'release/partition_utils.py','exec'),ns2)
loop_a=sp.csr_matrix(np.array([[2,1],[1,0]],dtype=np.float32))
original_parts,_=ns2['partition_graph'](loop_a,np.arange(2),1)
binary=loop_a.copy();binary.data[:]=1
np.testing.assert_array_equal(original_parts.toarray(),binary.toarray())
np.testing.assert_allclose(enhanced_support(binary).toarray(),[[4/3,1/3],[.5,1]],rtol=1e-6)
assert (loop_a@np.array([[1.],[2.]]))[0,0]==4
h=torch.arange(8,dtype=torch.float32).reshape(4,2).requires_grad_();st=sparse_tensor(enhanced_support(a))
y=concat_message(h,st);expected=torch.cat([torch.tensor(s)@h,h],1)
torch.testing.assert_close(y,expected);y.sum().backward()
torch.testing.assert_close(h.grad,(torch.tensor(s).sum(0)+1)[:,None].expand(-1,2))
# Dense independent layer oracle: per-row population variance, epsilon INSIDE sqrt.
torch.manual_seed(9);m=ClusterGCN(features=2,labels=3,hidden=5,layers=3,dropout=0).eval()
pre=torch.randn(4,4);got=m(pre,st)
z=pre
for i,w in enumerate(m.weights):
 if i:z=torch.cat([torch.tensor(s)@z,z],1)
 z=z@w
 if i<2:
  z=(z-z.mean(1,keepdim=True))/torch.sqrt(z.var(1,unbiased=False,keepdim=True)+1e-9)
  z=torch.relu(z*m.norms[i].weight+m.norms[i].bias)
torch.testing.assert_close(got,z,atol=2e-6,rtol=2e-6)
# Independent two-step TF1 Adam formula, including tiny gradients to expose epsilon placement.
p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64));opt=ReleaseAdam([p]);mv=np.zeros(2);vv=np.zeros(2);want=np.array([1.,2.])
for t,g in enumerate([np.array([1e-10,.2]),np.array([2e-10,-.3])],1):
 p.grad=torch.tensor(g);opt.step();mv=.9*mv+.1*g;vv=.999*vv+.001*g*g
 want-=.01*np.sqrt(1-.999**t)/(1-.9**t)*mv/(np.sqrt(vv)+1e-8)
np.testing.assert_allclose(p.detach().numpy(),want,atol=1e-12)
assert micro_f1([[0,1,-1]],[[1,1,0]])==2/3
# Test is a receiver/sender exclusion intervention, not merely a disjoint ID assertion.
d={'a':a,'x':np.arange(8,dtype=np.float32).reshape(4,2),'y':np.ones((4,3)),
   'masks':{'train':np.array([1,1,0,0],bool)}}
x=prepare(d,1);d2=dict(d,x=d['x'].copy(),y=d['y'].copy(),a=a.copy().tolil())
d2['x'][2:]=1e6;d2['y'][2:]=0;d2['a'][0,3]=100;d2['a']=d2['a'].tocsr();x2=prepare(d2,1)
np.testing.assert_array_equal(x['pre'],x2['pre']);np.testing.assert_array_equal(x['y'],x2['y']);assert (x['a']!=x2['a']).nnz==0
manifest=json.loads((LAB/'_sources_l089.json').read_text())
for f,sha in manifest['files'].items():assert hashlib.sha256((LAB/'sources/l089'/f).read_bytes()).hexdigest()==sha
assert preset('paper')==dict(hidden=2048,layers=5,epochs=400,num_parts=50,q=1,dropout=.2,lr=.01,partition_method='metis',partition_seed=1)
r={'status':'PASS','tests':['union restores cut edge and preserves local ID order','archived release normalization oracle','archived binary partition and raw self-loop cache oracle','concat forward and gradient','complete dense model oracle','TF1 Adam epsilon arithmetic','strict zero threshold micro-F1','heldout feature label edge intervention','source hashes','full recipe constants'],'tensorflow_forward_parity':'NOT_RUN'}
(LAB/'_verify_l089_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

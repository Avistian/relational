"""Independent arithmetic, real split components, and intervention checks for L083."""
import json,hashlib
from pathlib import Path
import networkx as nx
from relkit.graphsage_l083 import *
LAB=Path(__file__).resolve().parent;torch.set_num_threads(1)
manifest=json.loads((LAB/'_sources_l083.json').read_text())
for f,h in manifest['files'].items():assert hashlib.sha256((LAB/'sources/l083'/f).read_bytes()).hexdigest()==h
adj=[np.array([1,2]),np.array([0]),np.array([0])];mask=np.array([True,True,False])
eligible=eligible_neighbors(adj,mask);assert [a.tolist() for a in eligible]==[[1],[0],[]]
table=padded_adjacency(eligible,4,np.random.RandomState(0));assert table.tolist()==[[1]*4,[0]*4,[3]*4,[3]*4]
s=sample_support(torch.tensor([0]),table,[2,3],torch.Generator().manual_seed(0))
assert [v.tolist() for v in s]==[[0],[1,1],[0]*6]
a=torch.tensor([[2.,4.]]);b=torch.tensor([[[1.,3.],[5.,1.]]]);w=torch.eye(2)
actual=mean_concat(a,b,w,w);torch.testing.assert_close(actual,torch.tensor([[2.,4.,3.,2.]]))
torch.testing.assert_close(mean_concat(a,b.flip(1),w,w),actual)
# Independent NumPy oracle, two nonlinear layers, nontrivial weights, repeated support nodes.
torch.manual_seed(7);model=GraphSAGE(2,3,branch_dim=2).eval();x=torch.tensor([[2.,4.],[1.,3.],[5.,1.],[0.,0.]])
samples=[torch.tensor([0]),torch.tensor([1,2]),torch.tensor([0,2,0,1])]
h=[x[i].numpy() for i in samples]
for k,l in enumerate(model.layers):
    nxt=[]
    for hop in range(len(h)-1):
        mean=h[hop+1].reshape(len(h[hop]),2,-1).mean(1)
        val=np.concatenate([h[hop]@l.w_self.detach().numpy(),mean@l.w_neighbor.detach().numpy()],1)
        nxt.append(np.maximum(val,0) if k==0 else val)
    h=nxt
h=h[0]/np.sqrt(np.maximum((h[0]**2).sum(1,keepdims=True),1e-12))
expected=h@model.head.weight.detach().numpy().T+model.head.bias.detach().numpy()
np.testing.assert_allclose(model(x,samples,[2,2]).detach().numpy(),expected,rtol=2e-6,atol=1e-7)
data=load_ppi(LAB/'data/l083',manifest)
assert data['x'].shape==(56945,50) and data['y'].shape==(56944,121)
g=nx.Graph();g.add_nodes_from(range(56944))
for i,a in enumerate(data['adj']):g.add_edges_from((i,int(j)) for j in a)
components=list(nx.connected_components(g));counts={k:0 for k in data['masks']}
# Connected components need not equal tissue count: some tissues have disconnected pieces.
for comp in components:
    split=[k for k,m in data['masks'].items() if m[list(comp)].any()];assert len(split)==1
    counts[split[0]]+=1
induced=eligible_neighbors(data['adj'],data['masks']['train'])
assert all(data['masks']['train'][a].all() for a in induced)
config=dict(PAPER_CONFIG,epochs=1,branch_dim=8,fanouts=[2,3],batch_size=512)
m,r,table=fit_candidate(data,123,.01,config)
changed=dict(data,x=data['x'].clone(),y=data['y'].clone())
held=~data['masks']['train'];changed['y'][held]=1-changed['y'][held];changed['x'][:-1][held]=10000
m2,r2,_=fit_candidate(changed,123,.01,config)
for k,v in m.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)
assert r['trace']==r2['trace']
report={'status':'PASS','checks':['hashes','induced graph','zero sentinel','support ordering','mean arithmetic','neighbor permutation','independent two-layer NumPy oracle','PPI shapes','split-isolated connected components','full one-epoch held-out feature/label intervention'], 'components_by_split':counts,'original_tensorflow_parity':'NOT_RUN'}
(LAB/'_verify_l083_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

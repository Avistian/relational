"""Execute pinned original Python neural modules using an independent dense graph adapter.
This checks source algebra on modern PyTorch, NOT the historical DGL kernel/runtime.
"""
import ast,contextlib,copy,json,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from relkit.cvitkovic_l118 import *
P=Path(__file__).resolve().parent;S=P/'sources/l118'

def extracted(path,names,namespace):
    text=path.read_text();tree=ast.parse(text)
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)

class DenseGraph:
    def __init__(self,edges,batch):
        self.batch=batch;self.ndata={};self.n=len(batch)
        self.adjacency=torch.zeros(self.n,self.n)
        for u,v in edges.T.tolist():self.adjacency[v,u]+=1
    def local_var(self):
        g=copy.copy(self);g.ndata=self.ndata.copy();return g
    @contextlib.contextmanager
    def local_scope(self):
        old=self.ndata.copy()
        try:yield
        finally:self.ndata=old
    def in_degrees(self):return self.adjacency.sum(1)
    def update_all(self,*args):self.ndata['h']=self.adjacency.to(self.ndata['h'])@self.ndata['h']

def softmax_nodes(g,key):
    out=torch.zeros_like(g.ndata[key])
    for b in g.batch.unique():out[g.batch==b]=torch.softmax(g.ndata[key][g.batch==b],dim=0)
    return out

def sum_nodes(g,key):return torch.stack([g.ndata[key][g.batch==b].sum(0) for b in g.batch.unique()])
ns={'torch':torch,'th':torch,'nn':nn,'init':nn.init,'np':np,'activations':nn,'losses':nn,'BatchedDGLGraph':DenseGraph,'fn':types.SimpleNamespace(copy_src=lambda **k:None,sum=lambda **k:None),'softmax_nodes':softmax_nodes,'sum_nodes':sum_nodes}
extracted(S/'data/data_encoders.py',['EmbeddingInitializer'],ns)
extracted(S/'models/tabular/TabModelBase.py',['TabModelBase'],ns)
extracted(S/'models/tabular/TabMLP.py',['TabMLP'],ns)
extracted(S/'dgl_conv_0_3_1.py',['GraphConv'],ns)
dgl_ns=ns.copy()
extracted(S/'dgl_glob_0_3_1.py',['GlobalAttentionPooling'],dgl_ns)
ns['GAP']=dgl_ns['GlobalAttentionPooling']
extracted(S/'models/readouts.py',['GlobalAttentionPooling'],ns)
torch.manual_seed(11);port=CvitkovicGCN({'A':([4],2),'B':([],2)},{'A':0,'B':1},hidden=8,dropout=0)
original=[]
for name,cards in [('A',[4]),('B',[])]:
    row=ns['TabMLP'](writer=None,dataset_name=None,n_cont_features=2,cat_feat_origin_cards=[('f',4)] if cards else [],n_out=8,layer_sizes=[4.0],max_emb_dim=32,p_dropout=0.,one_hot_embeddings=False,drop_whole_embeddings=False,norm_class_name='Identity',norm_class_kwargs={},activation_class_name='SELU',activation_class_kwargs={})
    original.append(row)
    with torch.no_grad():
        for a,b in zip(port.encoders[name].parameters(),row.parameters()):b.copy_(a)
conv=ns['GraphConv'](8,8,activation=nn.SELU());pool=ns['GlobalAttentionPooling'](8,2,'SELU');head=nn.Linear(8,2)
conv.weight.data.copy_(port.weight);conv.bias.data.copy_(port.bias)
pool.gap.gate_nn.load_state_dict(port.gate.state_dict());pool.gap.feat_nn.load_state_dict(port.value.state_dict());head.load_state_dict(port.head.state_dict())
features={'A':(torch.tensor([[1],[2]]),torch.tensor([[1.,0.],[2.,1.]])),'B':(torch.empty(3,0,dtype=torch.long),torch.tensor([[.1,0.],[.4,0.],[.7,1.]]))};node_types=torch.tensor([0,1,1,0,1]);batch=torch.tensor([0,0,0,1,1]);edges=computation_edges(5,[(1,0),(2,0),(2,0),(4,3)]);graph=DenseGraph(edges,batch)
x=torch.zeros(5,8)
for i,t in enumerate(['A','B']):x[node_types==i]=original[i](features[t])
ref=head(pool(graph,conv(graph,x)));out=port(features,node_types,edges,batch,2)
error=float((out-ref).abs().max().detach());assert torch.allclose(out,ref,atol=1e-6,rtol=1e-5)
out.square().sum().backward();ref.square().sum().backward()
pairs=[(port.weight,conv.weight),(port.bias,conv.bias)]+list(zip(port.gate.parameters(),pool.gap.gate_nn.parameters()))+list(zip(port.value.parameters(),pool.gap.feat_nn.parameters()))+list(zip(port.head.parameters(),head.parameters()))
for t,row in zip(['A','B'],original):pairs+=list(zip(port.encoders[t].parameters(),row.parameters()))
grad=max(float((a.grad-b.grad).abs().max()) for a,b in pairs)
assert grad<1e-5
# Independent transitive closure oracle for every target on random directed multigraphs.
rng=np.random.default_rng(44);cases=0
for n in range(1,12):
 for repeat in range(12):
  e=[tuple(map(int,p)) for p in rng.integers(0,n,size=(2*n,2))];reach=np.eye(n,dtype=bool)
  for u,v in e:reach[u,v]=True
  for k in range(n):reach|=reach[:,k,None]&reach[None,k,:]
  for target in range(n):
   incoming=np.flatnonzero(reach[:,target]);wanted=np.flatnonzero(reach[incoming].any(0)).tolist()
   assert rdb_to_graph(n,e,target)[0]==wanted;cases+=1
# Training batch and node-order invariance in evaluation, no dropout.
port.eval();single=[]
for b in range(2):
 ids=torch.where(batch==b)[0];mapping={int(x):i for i,x in enumerate(ids)};es=[(mapping[int(u)],mapping[int(v)]) for u,v in edges.T if int(u) in mapping and int(v) in mapping]
 feats={};
 for t,(cat,cont) in features.items():
  local_batch=batch[node_types==port.type_ids[t]];mask=local_batch==b
  if mask.any():feats[t]=(cat[mask],cont[mask])
 single.append(port(feats,node_types[ids],torch.tensor(es).T,torch.zeros(len(ids),dtype=torch.long),1))
assert torch.allclose(port(features,node_types,edges,batch,2),torch.cat(single),atol=1e-6)
# Original fold helper execution proves identity on all released labeled IDs.
fold_ns={'np':np,'KFold':KFold,'train_test_split':train_test_split}
extracted(S/'data/utils.py',['train_val_split','five_fold_split_iter'],fold_ns)
info=json.loads((S/'data/homecreditdefaultrisk/homecreditdefaultrisk.db_info.json').read_text());splits=[]
for actual,(tv,test) in zip(released_folds(info['train_dp_ids']),fold_ns['five_fold_split_iter'](info['train_dp_ids'])):
 expected=(*fold_ns['train_val_split'](tv),test)
 for a,b in zip(actual,expected):assert np.array_equal(a,b)
 splits.append([len(a) for a in actual])
result={'status':'PASS','original_python_modules':['TabMLP','EmbeddingInitializer','DGL 0.3.1 GraphConv','DGL 0.3.1 GlobalAttentionPooling','release readout wrapper'],'graph_backend':'independent dense adapter, not historical DGL binary','forward_max_abs_error':error,'gradient_max_abs_error':grad,'random_extraction_cases':cases,'batch_separation':'PASS','full_labeled_id_fold_identity':'EXACT','fold_sizes_train_val_test':splits,'full_data_training':'NOT_RUN','historical_runtime':'NOT_CHECKED'}
(P/'_source_check_l121_results.json').write_text(json.dumps(result,indent=2));print(result)

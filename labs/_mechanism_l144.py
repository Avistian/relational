"""Differential source score/gradient and full neural fixture checks."""
import sys,json,tempfile
from pathlib import Path
import numpy as np
import torch
from relkit.context_l144 import fuse_scores,audit_cutoffs,keyed_map
from relkit.contextgnn_l144 import ContextGNN,ShallowRHSGNN,RHSEmbeddingMode

def mechanism_check(source, visible_namespace=None):
 sys.path.insert(0,str(source))
 import ast
 import relkit.contextgnn_l144 as visible
 original_source=(Path(source)/"contextgnn/nn/models/contextgnn.py").read_text()
 original_class=next(n for n in ast.parse(original_source).body if isinstance(n,ast.ClassDef) and n.name=="ContextGNN")
 namespace=dict(vars(visible) if visible_namespace is None else visible_namespace);exec(compile(ast.Module(body=[original_class],type_ignores=[]),str(source),"exec"),namespace);Original=namespace["ContextGNN"]
 # Isolate exact construct_logits operator, including its two learned offsets.
 torch.manual_seed(7)
 a=ContextGNN.__new__(ContextGNN);torch.nn.Module.__init__(a)
 a.head=torch.nn.Linear(4,1);a.lin_offset_idgnn=torch.nn.Linear(3,1);a.lin_offset_embgnn=torch.nn.Linear(3,1)
 b=Original.__new__(Original);torch.nn.Module.__init__(b)
 import copy
 b.head=copy.deepcopy(a.head);b.lin_offset_idgnn=copy.deepcopy(a.lin_offset_idgnn);b.lin_offset_embgnn=copy.deepcopy(a.lin_offset_embgnn)
 inputs=[torch.randn(2,3),torch.randn(2,4),torch.randn(3,4),torch.randn(5,3)]
 left=[x.clone().requires_grad_() for x in inputs];right=[x.clone().requires_grad_() for x in inputs]
 owners=torch.tensor([0,0,1]);items=torch.tensor([1,3,1])
 x=a.construct_logits(*left,owners,items);y=b.construct_logits(*right,owners,items)
 assert torch.equal(x,y);x.square().sum().backward();y.square().sum().backward()
 assert all(torch.equal(l.grad,r.grad) for l,r in zip(left,right))
 assert all(torch.equal(l.grad,r.grad) for l,r in zip(a.parameters(),b.parameters()))
 # Actual complete neural forward and backward, fixture data only.
 # Fixture-only pandas3 compatibility: explicit ns conversion and owned array.
 import relbench.modeling.graph as fixture_graph
 original_time_function=fixture_graph.to_unix_time
 fixture_graph.to_unix_time=lambda series:series.astype("datetime64[ns]").astype("int64").to_numpy(copy=True)//10**9
 from relbench.datasets.fake import FakeDataset
 from relbench.modeling.graph import make_pkey_fkey_graph,get_link_train_table_input
 from relbench.modeling.utils import get_stype_proposal
 from relbench.tasks.amazon import UserItemPurchaseTask
 from torch_frame.config import TextEmbedderConfig
 from torch_frame.testing.text_embedder import HashTextEmbedder
 from torch_geometric.loader import NeighborLoader
 from torch_geometric.utils.cross_entropy import sparse_cross_entropy
 from relbench.modeling.loader import SparseTensor
 from torch_geometric.seed import seed_everything
 seed_everything(7)
 with tempfile.TemporaryDirectory() as temp:
  ds=FakeDataset();db=ds.get_db();graph,stats=make_pkey_fkey_graph(db,get_stype_proposal(db),text_embedder_cfg=TextEmbedderConfig(text_embedder=HashTextEmbedder(8),batch_size=None),cache_dir=temp)
  task=UserItemPurchaseTask(ds);table=task.get_table('train');table.df[task.time_col]=table.df[task.time_col].astype('datetime64[ns]');inp=get_link_train_table_input(table,task)
  batch=next(iter(NeighborLoader(graph,num_neighbors=[8,4],time_attr='time',input_nodes=inp.src_nodes,input_time=inp.src_time,subgraph_type='bidirectional',batch_size=4,temporal_strategy='last')))
  kwargs=dict(data=graph,col_stats_dict=stats,dst_entity_table=task.dst_entity_table,num_nodes=inp.num_dst_nodes,num_layers=2,channels=8,embedding_dim=4,norm='layer_norm',torch_frame_model_kwargs={'channels':8,'num_layers':2})
  OMode=RHSEmbeddingMode
  model=ContextGNN(**kwargs,rhs_emb_mode=RHSEmbeddingMode.FUSION);ref=Original(**kwargs,rhs_emb_mode=OMode.FUSION);ref.load_state_dict(model.state_dict());model.eval();ref.eval()
  with torch.no_grad():v=model(batch,task.src_entity_table,task.dst_entity_table);w=ref(batch,task.src_entity_table,task.dst_entity_table)
  assert torch.allclose(v,w,atol=1e-6,rtol=1e-5)
  model.train();opt=torch.optim.Adam(model.parameters(),lr=.001);src,dst=SparseTensor(inp.dst_nodes[1])[batch[task.src_entity_table].input_id]
  out=model(batch,task.src_entity_table,task.dst_entity_table);loss=sparse_cross_entropy(out,torch.stack([src,dst]));loss.backward();assert torch.isfinite(loss);opt.step()
  # Preserve and reveal the upstream cache issue, not fix it inside reproduction.
  model.eval();cached=model.rhs_embedding().clone();model.rhs_embedding.lookup_embedding.weight.data.add_(1)
  stale=model.rhs_embedding();assert torch.equal(stale,cached)
  model.rhs_embedding._cached_rhs_embedding=None;fresh=model.rhs_embedding();assert not torch.equal(fresh,cached)
 fixture_graph.to_unix_time=original_time_function
 return {'status':'PASS','fusion_output_gradient_parity':'EXACT','neural_fixture_max_abs':float((v-w).abs().max()),'neural_fixture_loss':float(loss.detach()),'upstream_stale_rhs_cache':'CONFIRMED','fixture_is_paper_evidence':False}

if __name__=='__main__':
 result=mechanism_check(Path(__file__).resolve().parent/'sources/l144');(Path(__file__).parent/'_mechanism_l144_results.json').write_text(json.dumps(result,indent=2));print(result)

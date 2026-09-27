"""Pinned source structure and current-CPU neural output/gradient/update parity."""
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from relkit.rdl_l117 import Model
P=Path(__file__).resolve().parent;source=P/'sources/l117'
spec=importlib.util.spec_from_file_location('l117_original_model',source/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
canonical=ast.parse((P/'relkit/rdl_l117.py').read_text())
for name in ['nn.py','graph.py','model.py']:
 for node in ast.parse((source/name).read_text()).body:
  if not isinstance(node,(ast.ClassDef,ast.FunctionDef)):continue
  if node.name in ['LinkTrainTableInput','get_link_train_table_input']:continue
  current=next((n for n in canonical.body if isinstance(n,type(node)) and n.name==node.name),None)
  if current is None:continue
  if node.name=='Model':node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name!='forward_dst_readout']
  assert ast.dump(node,include_attributes=False)==ast.dump(current,include_attributes=False),node.name

torch.set_num_threads(1);torch.manual_seed(117);data=HeteroData();stats={}
for kind,values in [('drivers',[1.,2.,3.,4.]),('results',[3.,5.,9.,10.])]:
 d=Dataset(pd.DataFrame({'x':values}),col_to_stype={'x':stype.numerical}).materialize()
 data[kind].tf=d.tensor_frame;stats[kind]=d.col_stats;data[kind].batch=torch.tensor([0,1,0,1]);data[kind].n_id=torch.arange(4)
data['drivers'].seed_time=torch.tensor([20,30]);data['results'].time=torch.tensor([4,5,6,7])
data['results','to','drivers'].edge_index=torch.tensor([[0,1,2,3],[0,1,0,1]])
data['drivers','rev_to','results'].edge_index=data['results','to','drivers'].edge_index.flip(0)
for kind in data.node_types:data[kind].num_sampled_nodes=[2,1,1]
for relation in data.edge_types:data[relation].num_sampled_edges=[2,2]
m=Model(data,stats,2,128,1,'sum','batch_norm');ref=mod.Model(data,stats,2,128,1,'sum','batch_norm');ref.load_state_dict(m.state_dict());m.eval();ref.eval()
p=m(data,'drivers');q=ref(data,'drivers');torch.testing.assert_close(p,q,rtol=0,atol=0)
target=torch.tensor([[3.],[7.]]);loss=(p-target).abs().mean();other=(q-target).abs().mean();loss.backward();other.backward()
for a,b in zip(m.parameters(),ref.parameters()):
 if a.grad is None or b.grad is None:assert a.grad is None and b.grad is None
 else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
opt=torch.optim.Adam(m.parameters(),lr=.005);ropt=torch.optim.Adam(ref.parameters(),lr=.005);opt.step();ropt.step()
for a,b in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
r={'status':'PASS','source_definitions':'AST_EXACT for included upstream classes/functions','cpu_model_outputs':'EXACT','cpu_gradients':'EXACT','cpu_adam_update':'EXACT','scope':'Four rows per table, current local runtime. Full-data pinned GPU replay is recorded separately.','torch':torch.__version__}
(P/'_source_check_l117_results.json').write_text(json.dumps(r,indent=2));print(r)

"""Independent Frame/PyG references for every exported row and graph messages."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch,torch_frame
from torch_frame.nn.models import ResNet
from torch_geometric.nn import SAGEConv
from relkit.frame_l125 import *
from _check_l125 import check_numeric,check_categories,check_alignment,check_fit
P=Path(__file__).resolve().parent;E=P/'evidence/l125'
mutations={}
for name,check,mutant in [
 ('omit_centering',check_numeric,lambda x,m,s,w,b:(torch.nan_to_num(x)/s).unsqueeze(-1)*w+b),
 ('omit_column_offsets',check_categories,lambda x,c:torch.where(x<0,0,x+1)),
 ('ignore_row_ids',check_alignment,lambda ids,x,q:x[:len(q)]),
 ('fit_query_rows',check_fit,lambda tr,q,t:fit_and_convert(pd.concat([tr,q],ignore_index=True),q,t))]:
 try:check(mutant)
 except (AssertionError,ValueError,IndexError):mutations[name]='REJECTED'
 else:raise AssertionError('Surviving mutation: '+name)
# A row-local encoder cannot change a different row; a GNN can.
s=torch.tensor([[2.,0.],[4.,0.],[8.,2.]],requires_grad=True);edges=torch.tensor([[0,1,2],[0,0,1]])
expected=torch.tensor([[3.,0.],[8.,2.],[0.,0.]])
torch.testing.assert_close(mean_messages(s,edges,3),expected)
head=CourseGraphHead(2);target=torch.tensor([[1.,1.],[2.,1.],[0.,1.]])
conv=SAGEConv((2,2),2,aggr='mean');conv.lin_l.weight.data.copy_(head.neighbor.weight);conv.lin_l.bias.data.copy_(head.root.bias);conv.lin_r.weight.data.copy_(head.root.weight)
torch.testing.assert_close(head(target,s,edges,torch.arange(3)),head.head(torch.relu(conv((s,target),edges))).flatten())
# Verify installed load-bearing Frame files against the archived release.
root=Path(torch_frame.__file__).parent
for rel in ['nn/models/resnet.py','nn/encoder/stype_encoder.py','nn/encoder/stypewise_encoder.py']:
 assert (root/rel).read_bytes()==(P/'sources/l125/frame/torch_frame'/rel).read_bytes(),rel
# Recompute all vectors, using the upstream model forward rather than our tokens.
tables=load_f1_archive((E/'f1-db.zip').read_bytes());models,frames,exports,counts=encode_f1_tables(tables)
saved=np.load(E/'encoded-reg.npz',allow_pickle=False);max_error=0;total=0
for name,model in models.items():
 ref=ResNet(channels=8,out_channels=8,num_layers=2,col_stats=model.encoder.col_stats,col_names_dict=model.encoder.col_names_dict,stype_encoder_dict=encoder_recipe(),dropout_prob=0.).eval()
 ref.load_state_dict(model.state_dict());tf=frames[name]
 with torch.no_grad():z=torch.cat([ref(tf[i:i+1024]) for i in range(0,len(tf),1024)])
 np.testing.assert_allclose(z.numpy(),saved[name+'_vectors'],rtol=1e-5,atol=1e-6)
 assert saved[name+'_ids'].tolist()==tables[name][F1_KEYS[name]].tolist()
 max_error=max(max_error,float(np.abs(z.numpy()-saved[name+'_vectors']).max()));total+=len(z)
assert len({id(m.backbone[0].lin1.weight) for m in models.values()})==9
r={'status':'PASS','mutation_checks':mutations,'full_data_original_frame_rows':total,'maximum_output_error':max_error,'installed_frame_sources':'EXACT','sageconv_message_parity':'PASS','independent_table_parameters':9,'paper_result':'NOT_RUN'}
(P/'_audit_l125_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

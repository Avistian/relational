"""Execute unmodified extracted release primitives on local counterexamples.
Current dependency behavior is not proof of the authors' historical runtime.
"""
import ast,copy,hashlib,inspect,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv,MessagePassing
from relbench.modeling.nn import HeteroGraphSAGE
import importlib.metadata as md
P=Path(__file__).resolve().parent;S=P/'sources/b21'
text=(S/'utils.py').read_text();tree=ast.parse(text)
names=['WeightedSAGEConv','AttackableHeteroGraphSAGE','_get_num_local_nodes','sample_candidate_support_per_child_vectorized','_build_edge_lookup','apply_forward_rewirings','select_unique_relation_child']
ns=dict(torch=torch,nn=nn,np=np,copy=copy,MessagePassing=MessagePassing,HeteroConv=HeteroConv)
for name in names:
    node=next(x for x in tree.body if getattr(x,'name','')==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(S/'utils.py'),'exec'),ns)
out=dict(scope='Unmodified release primitives with current installed dependencies',versions={n:md.version(n) for n in ['torch','torch-geometric','relbench']})
torch.set_num_threads(1)
rel=('events','f2p_owner','accounts');rev=('accounts','rev_f2p_owner','events')
e=torch.tensor([[0,1,2,3,4,5],[0,0,1,2,1,2]])
batch=HeteroData();batch['events'].num_nodes=6;batch['accounts'].num_nodes=4;batch[rel].edge_index=e;batch[rev].edge_index=e.flip(0)
batch['accounts'].time=torch.tensor([0,1,2,12]);batch['events'].time=torch.tensor([3,4,5,6,7,8])
torch.manual_seed(0);candidate,mask,support=ns['sample_candidate_support_per_child_vectorized'](batch,rel,3)
future=[(int(c),int(p)) for c,p in candidate[:,~mask].T if int(p)==3]
out['future_candidate_pairs']=future;assert future
selected=[dict(relation=rel,child=0,old_dst=0,new_dst=2)]
changed=ns['apply_forward_rewirings'](batch,rel,selected,rev)
out['assignment_after_one_edit']=changed.edge_index_dict[rel][1].tolist()
out['underlying_edge_store_after_one_edit']=changed[rel].edge_index[1].tolist()
out['requested_edit_materialized']=int(changed.edge_index_dict[rel][1,0])==2
out['zero_budget_selected_count']=len(ns['select_unique_relation_child'](selected,0))
assert out['zero_budget_selected_count']==1
try:
    SEEDS=[39];rows={}
    for seed in [SEEDS]:rows[seed]=[]
except TypeError as error:out['literal_seed_loop_error']=str(error)
checks=[]
for seed in [0,1,2]:
    torch.manual_seed(seed)
    base=HeteroGraphSAGE(['events','accounts'],[rel,rev],4,aggr='sum',num_layers=2).eval()
    wrapper=ns['AttackableHeteroGraphSAGE'](['events','accounts'],[rel,rev],4,aggr='sum',num_layers=2).eval()
    info=wrapper.load_state_dict(base.state_dict(),strict=False)
    x={'events':torch.randn(6,4),'accounts':torch.randn(4,4)};edges={rel:e,rev:e.flip(0)}
    a=base(x,edges);b=wrapper(x,edges)
    error=max(float((a[k]-b[k]).abs().max().detach()) for k in a)
    # Controlled diagnostic repair only: zero the newly introduced root biases.
    with torch.no_grad():
        for name,param in wrapper.named_parameters():
            if name in info.missing_keys:param.zero_()
    repaired=wrapper(x,edges)
    repair_error=max(float((a[k]-repaired[k]).abs().max().detach()) for k in a)
    checks.append(dict(seed=seed,missing_keys=info.missing_keys,unexpected_keys=info.unexpected_keys,max_clean_embedding_error=error,zero_new_bias_error=repair_error))
assert all(x['max_clean_embedding_error']>1e-5 for x in checks)
out['clean_conversion']=checks
source=inspect.getsource(HeteroGraphSAGE);(S/'installed-HeteroGraphSAGE.py').write_text(source)
out['installed_base_sha256']=hashlib.sha256(source.encode()).hexdigest()
(P/'evidence/b21/source-probes.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

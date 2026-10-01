"""Small full neural BPR update plus copied-weight source output/gradient parity."""
import copy,importlib.util,json
from pathlib import Path
import torch,pandas as pd,torch_frame
from torch_geometric.data import HeteroData
from torch_frame.data import Dataset
from relkit.recommendation_model_l153 import Model

def mechanism_check(source,model_class=Model):
    torch.set_num_threads(1);torch.manual_seed(153)
    graph=HeteroData();stats={}
    for name,values in [('sites',[0.,1.,3.]),('sponsors',[1.,2.,4.,7.])]:
        ds=Dataset(pd.DataFrame({'value':values}),col_to_stype={'value':torch_frame.numerical}).materialize()
        graph[name].tf=ds.tensor_frame;stats[name]=ds.col_stats;graph[name].num_nodes=len(values)
        graph[name].time=torch.zeros(len(values),dtype=torch.long);graph[name].num_sampled_nodes=[len(values),0,0];graph[name].n_id=torch.arange(len(values));graph[name].batch=torch.zeros(len(values),dtype=torch.long);graph[name].seed_time=torch.tensor([10,10])
    edges=torch.tensor([[0,0,1,1,2,2],[0,1,1,2,2,3]])
    graph['sites','uses','sponsors'].edge_index=edges;graph['sponsors','rev_uses','sites'].edge_index=edges.flip(0)
    for edge in graph.edge_types:graph[edge].num_sampled_edges=[6,0]
    kwargs=dict(data=graph,col_stats_dict=stats,num_layers=2,channels=8,out_channels=8,aggr='sum',norm='layer_norm',shallow_list=['sponsors'])
    model=model_class(**kwargs);spec=importlib.util.spec_from_file_location('original_l153',Path(source)/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);ref=module.Model(**kwargs);ref.load_state_dict(model.state_dict())
    model.eval();ref.eval()
    def objective(m):
        users=m(graph,'sites');items=m(graph,'sponsors')
        scores=users@items.T;positive=scores.diag().reshape(-1,1)
        return scores,torch.nn.functional.softplus(scores-positive).mean()
    left,loss=objective(model);right,ref_loss=objective(ref);assert torch.allclose(left,right,atol=1e-6,rtol=1e-5)
    loss.backward();ref_loss.backward();parameters=0
    for (name,a),(_,b) in zip(model.named_parameters(),ref.named_parameters()):
        if a.grad is not None:
            assert torch.isfinite(a.grad).all(),name;assert torch.allclose(a.grad,b.grad,atol=1e-6,rtol=1e-5),name;parameters+=a.numel()
    before=model.embedding_dict['sponsors'].weight.detach().clone();torch.optim.Adam(model.parameters(),lr=.001).step();assert not torch.equal(before,model.embedding_dict['sponsors'].weight)
    return dict(status='PASS',output_max_abs=float((left-right).abs().max().detach()),loss=float(loss.detach()),finite_gradient_parameters=parameters,shallow_embedding_update='PASS',scope='SYNTHETIC_MECHANISM_ONLY')

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=mechanism_check(p/'sources/l153');(p/'_mechanism_l153_results.json').write_text(json.dumps(r,indent=2));print(r)

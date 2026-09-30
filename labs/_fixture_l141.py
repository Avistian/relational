"""Synthetic whole-model forward/backward; no paper-result claim."""
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData

def neural_fixture(Model,route_builder):
    torch.set_num_threads(1);torch.manual_seed(141);data=HeteroData();stats={}
    for name,values in [('drivers',[1.,2.,3.,4.]),('results',[3.,5.,9.,10.]),('constructors',[2.,7.,11.,16.])]:
        frame=Dataset(pd.DataFrame({'x':values}),col_to_stype={'x':stype.numerical}).materialize()
        data[name].tf=frame.tensor_frame;stats[name]=frame.col_stats;data[name].num_nodes=4
        data[name].batch=torch.tensor([0,1,0,1]);data[name].n_id=torch.arange(4)
    data['drivers'].seed_time=torch.tensor([864000,1728000]);data['results'].time=torch.tensor([777600,1555200,691200,1468800])
    for dst,fk in [('drivers','driverId'),('constructors','constructorId')]:
        edge=torch.tensor([[0,1,2,3],[0,1,0,1]])
        data['results','f2p_'+fk,dst].edge_index=edge;data[dst,'rev_f2p_'+fk,'results'].edge_index=edge.flip(0)
    model=Model(data,stats,1,128,1,'sum','batch_norm',atomic_routes=route_builder(data.edge_types),num_heads=4)
    model.train();pred=model(data,'drivers').view(-1);assert pred.shape==(2,)
    loss=torch.nn.functional.l1_loss(pred,torch.tensor([3.,5.]));loss.backward();report={}
    for name,part in [('encoder',model.encoder),('time',model.temporal_encoder),('gnn',model.gnn),('head',model.head)]:
        grads=[p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm=sum(float(g.square().sum()) for g in grads)**.5;assert norm>0
        report[name]=norm
    return dict(status='PASS',prediction=pred.detach().tolist(),loss=float(loss.detach()),gradient_norms=report,scope='finite synthetic fixture; real missing-value gradients audited separately')
if __name__=='__main__':
    from relkit.relgnn_l141 import RelGNN_Model,get_atomic_routes
    print(neural_fixture(RelGNN_Model,get_atomic_routes))

"""Small complete RDL forward/backward, separate from clinical trial training."""
import copy
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData

def neural_fixture(Model):
    torch.set_num_threads(1);torch.manual_seed(139)
    data=HeteroData();stats={}
    for kind,values in [('studies',[1.,2.,3.,4.]),('facilities_studies',[3.,5.,9.,10.]),('facilities',[2.,7.,11.,16.])]:
        frame=Dataset(pd.DataFrame({'x':values}),col_to_stype={'x':stype.numerical}).materialize()
        data[kind].tf=frame.tensor_frame;stats[kind]=frame.col_stats
        data[kind].batch=torch.tensor([0,1,0,1]);data[kind].n_id=torch.arange(4);data[kind].num_sampled_nodes=[2,1,1]
    data['studies'].seed_time=torch.tensor([864000,1728000]);data['facilities_studies'].time=torch.tensor([777600,1555200,691200,1468800])
    for src,dst in [('facilities_studies','studies'),('facilities_studies','facilities')]:
        edge=torch.tensor([[0,1,2,3],[0,1,0,1]])
        data[src,'to',dst].edge_index=edge;data[dst,'rev_to',src].edge_index=edge.flip(0)
    for kind in data.edge_types:data[kind].num_sampled_edges=[2,2]
    model=Model(data,stats,2,128,1,'mean','batch_norm');model.train()
    logits=model(data,'studies').view(-1);assert logits.shape==(2,)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.]));loss.backward()
    for name,part in [('encoder',model.encoder),('gnn',model.gnn),('head',model.head)]:
        gradients=[p.grad for p in part.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.square().sum()) for g in gradients)>0,name
    return dict(status='PASS',logits=logits.detach().tolist(),loss=float(loss.detach()),scope='Synthetic full-stack forward/backward; not clinical trial model training'),data,stats,model

if __name__=='__main__':
    from relkit.rdl_l117 import Model
    print(neural_fixture(Model)[0])

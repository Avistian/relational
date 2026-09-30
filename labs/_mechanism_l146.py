"""Independent relation-mean oracle and actual model-gradient fixture."""
import numpy as np
import torch

def check_mean(fn):
    x=torch.tensor([[2.,1.],[4.,3.],[8.,5.]],requires_grad=True)
    edge=torch.tensor([[0,1,1],[2,2,0]])
    y=fn(x,edge,3)
    np.testing.assert_allclose(y.detach(),[[4,3],[0,0],[3,2]])
    y.sum().backward();np.testing.assert_allclose(x.grad,[[.5,.5],[1.5,1.5],[0,0]])
    assert torch.equal(fn(x,torch.empty((2,0),dtype=torch.long),3),torch.zeros_like(x))

if __name__=='__main__':
    from relkit.gnn_l146 import relation_mean
    check_mean(relation_mean)
    print('PASS: mean aggregation and exact gradients')

def full_fixture(source_root,namespace=None):
    import copy,pandas as pd,torch_frame
    if namespace is None:
        from relkit.relgt_course_l146 import RelGT,EncoderLayer
        from relkit.gnn_l146 import CourseGNN
    else:
        RelGT=namespace['RelGT'];EncoderLayer=namespace['EncoderLayer'];CourseGNN=namespace['CourseGNN']
    from _mechanism_l145 import original_modules
    torch.set_num_threads(1);torch.manual_seed(146)
    frame=torch_frame.data.Dataset(pd.DataFrame({'x':[1.,2.,3.,4.,5.,6.]}),col_to_stype={'x':torch_frame.numerical});frame.materialize()
    columns={'row':frame.tensor_frame.col_names_dict};stats={'row':frame.col_stats};type_map={'row':0}
    grouped={'grouped_tfs':{0:frame.tensor_frame},'grouped_indices':{0:list(range(6))},'flat_batch_idx':[0,0,0,1,1,1],'flat_nbr_idx':[0,1,2,0,1,2]}
    b=dict(neighbor_types=torch.zeros(2,3,dtype=torch.long),node_indices=torch.tensor([0,3]),neighbor_hops=torch.tensor([[0,1,2],[0,1,2]]),neighbor_times=torch.tensor([[0.,2.,3.],[0.,1.,4.]]),**grouped)
    b.update(edge_index=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]]),batch=torch.tensor([0,0,0,1,1,1]),relation=torch.zeros(8,dtype=torch.long))
    g=CourseGNN(columns,stats,type_map,1,width=16);out=g(copy.deepcopy(b));assert out.shape==(2,);out.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in g.parameters())
    kw=dict(num_nodes=6,max_neighbor_hop=3,node_type_map=type_map,col_names_dict=columns,col_stats_dict=stats,local_num_layers=1,channels=16,out_channels=1,global_dim=8,heads=4,ff_dropout=.3,attn_dropout=.3,conv_type='full',num_centroids=8,sample_node_len=3)
    args=tuple(b[k] for k in ['neighbor_types','node_indices','neighbor_hops','neighbor_times'])+(grouped,)
    extra={k:b[k] for k in ['edge_index','batch']}
    with original_modules(source_root) as Original:
        a=RelGT(**kw);reference=Original(**kw);reference.load_state_dict(a.state_dict())
        torch.manual_seed(77);pa=a(*copy.deepcopy(args),**extra);pa.square().sum().backward()
        torch.manual_seed(77);pb=reference(*copy.deepcopy(args),**extra);pb.square().sum().backward()
        torch.testing.assert_close(pa,pb,rtol=1e-6,atol=1e-6);count=0
        for (na,va),(nb,vb) in zip(a.named_parameters(),reference.named_parameters()):
            assert na==nb
            if va.grad is not None:torch.testing.assert_close(va.grad,vb.grad,rtol=1e-6,atol=1e-6);count+=1
        a.eval();before={k:v.clone() for k,v in a.state_dict().items()}
        p1=a(*copy.deepcopy(args),**extra);p2=a(*copy.deepcopy(args),**extra);torch.testing.assert_close(p1,p2,atol=0,rtol=0)
        assert all(torch.equal(v,a.state_dict()[k]) for k,v in before.items())
    layer=EncoderLayer(16,32,.3,.3,4).eval();x=torch.randn(2,3,16)
    torch.testing.assert_close(layer(x),layer(x),atol=0,rtol=0)
    return dict(status='PASS',gnn_forward_backward='PASS',relgt_training_original_max_error=float((pa-pb).abs().max().detach()),gradient_tensors=count,deterministic_eval='PASS',eval_state_frozen='PASS')

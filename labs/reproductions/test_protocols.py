"""Scientific boundaries: grouped categorical corruption, gradients, and source parity."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
from scarf import ScarfEncoder, ScarfProjection, scarf_objective, corrupt_columns, paper_config


def test_scarf_architecture_and_loss():
    f=ScarfEncoder(9);g=ScarfProjection()
    assert len([m for m in f.modules() if isinstance(m,nn.Linear)])==4
    assert all(m.out_features==256 for m in f.modules() if isinstance(m,nn.Linear))
    assert len([m for m in g.modules() if isinstance(m,nn.Linear)])==2
    a=torch.eye(3,requires_grad=True);b=torch.eye(3,requires_grad=True)
    loss=scarf_objective(a,b)
    expected=torch.log(torch.tensor((1+2/np.e)/3))
    assert torch.allclose(loss,expected)
    loss.backward();assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0
    c=paper_config();assert (c['width'],c['pre_epochs'],c['epochs'],c['patience'],c['batch'])==(256,1000,200,3,128)


def test_original_column_corruption():
    # One numeric feature and one three-level categorical feature: one donor per original column.
    x=torch.tensor([[1.,1.,0.,0.],[2.,0.,1.,0.],[3.,0.,0.,1.]])
    changed,mask=corrupt_columns(x,x,[slice(0,1),slice(1,4)],1.,torch.Generator().manual_seed(8))
    assert torch.equal(changed[:,1:].sum(1),torch.ones(3))
    assert mask.shape==(3,2) and mask.all()
    clean,_=corrupt_columns(x,x,[slice(0,1),slice(1,4)],0.,torch.Generator().manual_seed(8))
    assert torch.equal(clean,x)

if __name__=='__main__':
    test_scarf_architecture_and_loss();test_original_column_corruption();print('SCARF protocol tests passed')


def test_subtab_full_forward_gradient_and_update():
    from subtab import SubTabMNIST,subtab_joint,subtab_views
    root=Path(__file__).resolve().parents[1]/'sources/l072'
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    reference=module('subtab_models',root/'utils__model_utils.py')
    loss_module=module('subtab_losses',root/'utils__loss_functions.py')
    options=dict(dims=[784,784,784],shallow_architecture=True,n_subsets=4,overlap=.75,isBatchNorm=False,
                 isDropout=False,normalize=True,p_norm=2,reconstruction=True,reconstruct_subset=False,
                 batch_size=4,tau=.1,device='cpu',cosine_similarity=False,contrastive_loss=True,distance_loss=True)
    torch.manual_seed(9);a=SubTabMNIST();b=reference.AEWrapper(options)
    pairs=list(zip(a.parameters(),b.parameters()));assert all(x.shape==y.shape for x,y in pairs)
    with torch.no_grad():
        for x,y in pairs:y.copy_(x)
    x=torch.randn(8,343);target=torch.rand(8,784)
    za,ha,ra=a(x);zb,hb,rb=b(x)
    for u,v in [(za,zb),(ha,hb),(ra,rb)]:assert torch.allclose(u,v,atol=1e-6)
    la=subtab_joint(za,ra,target);lb=loss_module.JointLoss(options)(zb,rb,target)[0]
    assert torch.allclose(la,lb,atol=1e-5)
    oa=torch.optim.AdamW(a.parameters(),lr=.001,eps=1e-7);ob=torch.optim.AdamW(b.parameters(),lr=.001,eps=1e-7)
    la.backward();lb.backward()
    for u,v in pairs:assert torch.allclose(u.grad,v.grad,atol=2e-6)
    oa.step();ob.step()
    for i,(u,v) in enumerate(pairs):
        assert torch.allclose(u,v,atol=2e-6), (i,float((u-v).abs().max()))
    raw=np.arange(5*784).reshape(5,784)
    views=subtab_views(raw,np.random.RandomState(57),False)
    assert len(views)==4 and all(v.shape==(5,343) for v in views)
    for view,(a,b) in zip(views,[(0,343),(49,392),(245,588),(441,784)]):
        assert np.array_equal(view,raw[:,a:b]), 'Official entrypoints disable ALL evaluation noise'
    train_views=subtab_views(raw,np.random.RandomState(57),True)
    assert any(not np.array_equal(view,raw[:,a:b]) for view,(a,b) in zip(train_views,[(0,343),(49,392),(245,588),(441,784)]))
    print('SubTab full forward/loss/gradient/AdamW update parity passed')

if __name__=='__main__':test_subtab_full_forward_gradient_and_update()


def test_carte_release_loader_and_ensemble():
    import sys
    from carte import load_carte_sources
    from torch_geometric.data import Data
    CARTERegressor,_,_=load_carte_sources()
    checkpoint=Path(__file__).resolve().parents[1]/'data/l074/kg_pretrained.pt'
    graph=Data(x=torch.randn(3,300),edge_index=torch.tensor([[0,0,1,2],[1,2,1,2]]),edge_attr=torch.randn(4,300),y=torch.tensor([1.]))
    est=CARTERegressor(num_model=15,pretrained_model_path=str(checkpoint))
    est.X_=[graph]*32;est.y_=np.arange(32);est.device_=torch.device('cpu');est._set_task_specific_settings()
    splits=est._set_train_valid_split()
    assert len(splits)==15 and all(len(a)==25 and len(b)==7 and not set(a)&set(b) for a,b in splits)
    loaded=est._load_model();est.load_pretrain=False;random=est._load_model()
    for a,b in zip(loaded.ft_base.initial_x.parameters(),random.ft_base.initial_x.parameters()):assert torch.equal(a,b)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for key,value in loaded.ft_base.read_out_block.state_dict().items():
        assert torch.equal(value,state['ft_base.read_out_block.'+key])
    assert all(not p.requires_grad for p in loaded.ft_base.read_out_block.parameters())
    linear=[m for m in loaded.modules() if isinstance(m,nn.Linear)]
    assert [(m.in_features,m.out_features) for m in linear[-3:]]==[(300,150),(150,75),(75,1)]
    print('CARTE full head, partial checkpoint loading, frozen block and 15-split checks passed')

if __name__=='__main__':test_carte_release_loader_and_ensemble()


def test_paper_era_carte():
    from carte_compat import scatter_sum,load_paper_carte
    src=torch.randn(8,5,requires_grad=True);index=torch.tensor([0,0,1,1,3,3,3,4])
    expected=torch.nn.functional.one_hot(index,5).float().T@src
    actual=scatter_sum(src,index)
    assert torch.equal(actual,expected)
    actual.square().sum().backward();assert torch.allclose(src.grad,2*expected[index])
    checkpoint=Path(__file__).resolve().parents[1]/'data/l074/kg_pretrained.pt'
    CARTERegressor,_,_=load_paper_carte(checkpoint=checkpoint)
    from torch_geometric.data import Data,Batch
    graph=Data(x=torch.randn(3,300),edge_index=torch.tensor([[0,0,1,2],[1,2,1,2]]),edge_attr=torch.randn(4,300),y=torch.tensor([1.]))
    est=CARTERegressor(num_model=15)
    est.X_=[graph]*32;est.y_=np.arange(32);est.device_=torch.device('cpu');est._set_task_specific_settings()
    model=est._load_model();output=model(Batch.from_data_list([graph,graph]))
    assert output.shape==(2,1) and torch.isfinite(output).all()
    assert len(est._set_train_valid_split())==15
    print('June 2024 CARTE full forward and native scatter output/gradient checks passed')

if __name__=='__main__':test_paper_era_carte()

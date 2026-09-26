"""Behavioral red/green probes; functions are injectable for live notebook tasks."""
import argparse,copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def check_loss(fn):
    logits=torch.tensor([[2.,-1.],[-1.,2.],[.7,-.3],[3.,-2.]],requires_grad=True)
    y=torch.tensor([0,1,1,0]);idx=torch.tensor([2,0]);p=logits.log_softmax(1)
    actual=fn(p,y,idx);expected=(-p[2,1]-p[0,0])/2
    torch.testing.assert_close(actual,expected)
    changed=y.clone();changed[torch.tensor([1,3])]=1-changed[torch.tensor([1,3])]
    torch.testing.assert_close(fn(p,changed,idx),actual,rtol=0,atol=0)
    g=torch.autograd.grad(actual,logits)[0]
    assert torch.count_nonzero(g[[1,3]])==0,'Held-out output rows receive supervised gradients'
    assert torch.count_nonzero(g[[0,2]])>0

def check_step(fn):
    class Tiny(nn.Module):
        def __init__(self):super().__init__();self.fc=nn.Linear(2,2)
        def forward(self,x,adj):return self.fc(adj@x).log_softmax(1)
    torch.manual_seed(13);model=Tiny();reference=copy.deepcopy(model)
    x=torch.tensor([[1.,2.],[3.,-1.],[-2.,1.]]);adj=torch.eye(3);y=torch.tensor([0,1,0]);idx=torch.tensor([2,0])
    opt=torch.optim.SGD(model.parameters(),lr=.1);refopt=torch.optim.SGD(reference.parameters(),lr=.1)
    before=copy.deepcopy(model.state_dict());model.eval()
    for _ in range(2):
        reference.train();refopt.zero_grad();expected=F.nll_loss(reference(x,adj)[idx],y[idx]);expected.backward();refopt.step()
        loss=fn(model,opt,x,adj,y,idx)
        assert model.training,'Training mode was not restored'
        torch.testing.assert_close(torch.as_tensor(loss),expected.detach())
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items()),'Parameters never changed'

def check_seed(fn):
    # Noncontiguous global IDs; first two local rows are seeds, rest context.
    n_id=torch.tensor([4,1,5,0]);y=torch.tensor([1,1,1,0,0,1]);b=2
    logits=torch.tensor([[2.,-1.],[-2.,1.],[4.,-4.],[-3.,3.]],requires_grad=True);p=logits.log_softmax(1)
    expected=(-p[0,0]-p[1,1])/2;actual=fn(p,y,n_id,b);torch.testing.assert_close(actual,expected)
    changed=y.clone();changed[[0,2,3,5]]=1-changed[[0,2,3,5]]
    torch.testing.assert_close(fn(p,changed,n_id,b),actual,rtol=0,atol=0)
    g=torch.autograd.grad(actual,logits)[0];assert torch.count_nonzero(g[b:])==0
    # Reordering context cannot change the supervised population.
    order=torch.tensor([0,1,3,2]);torch.testing.assert_close(fn(p[order],y,n_id[order],b),actual)

def broken_loss(p,y,idx):return F.nll_loss(p,y)
def broken_step(model,opt,x,adj,y,idx):
    model.train();opt.zero_grad();loss=F.nll_loss(model(x,adj)[idx],y[idx]);loss.backward()
    return loss.detach() # injected omission: no optimizer.step()
def broken_seed(p,y,n_id,b):return F.nll_loss(p[:b],y[:b])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--red',action='store_true');args=a.parse_args();torch.set_num_threads(1)
    if args.red:
        caught=[]
        for name,check,fn in [('held_out_labels',check_loss,broken_loss),('missing_step',check_step,broken_step),('global_ids',check_seed,broken_seed)]:
            try:check(fn)
            except AssertionError as e:caught.append(name);print('RED expected:',name,str(e).splitlines()[0])
        assert len(caught)==3,'A fault escaped detection'
        Path(__file__).with_name('_red_l116_results.json').write_text(json.dumps({'status':'PASS','expected_failures':caught},indent=2))
    else:
        from relkit.debug_l116 import training_loss,train_step,seed_loss
        for check,fn in [(check_loss,training_loss),(check_step,train_step),(check_seed,seed_loss)]:check(fn)
        r={'status':'PASS','behavioral_probes':['train-only loss and gradient isolation','two exact optimizer updates and train mode','seed/global ID and context-label independence']}
        Path(__file__).with_name('_check_l116_results.json').write_text(json.dumps(r,indent=2));print(r)

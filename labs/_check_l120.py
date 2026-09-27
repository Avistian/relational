"""Behavioral exit gates; each accepts the learner's live function."""
import json
from pathlib import Path
import torch

def check_keys(fn):
    assert fn([90,10],[10,None,90]).tolist()==[[0,2],[1,0]], 'Keys are not row offsets'
    for pk,fk in [([1,1],[1]),([1],[2])]:
        try:fn(pk,fk)
        except ValueError:pass
        else:raise AssertionError('Reject duplicate primary or dangling foreign keys')

def check_visibility(fn):
    assert fn([2,5,9],[3,8,9],7).tolist()==[True,False,False], 'Both clocks must be legal'
    assert fn([7,7],[7,8],7).tolist()==[True,False], 'Boundary is inclusive'

def check_roots(fn):
    from torch_geometric.data import HeteroData, Batch
    graphs=[]
    for n,r in [(3,2),(2,0)]:
        g=HeteroData();g['person'].x=torch.zeros(n,1);g['person'].root_mask=torch.arange(n)==r;graphs.append(g)
    b=Batch.from_data_list(graphs)
    assert fn(b).tolist()==[2,3], 'Query seeds are not necessarily a global prefix'

def check_loss(fn):
    p=torch.tensor([9.,2.,8.,4.],requires_grad=True)
    loss=fn(p,torch.tensor([1,3]),torch.tensor([1.,6.]),torch.tensor([4,7]),7)
    assert loss.item()==1.5;loss.backward()
    assert p.grad.tolist()==[0.,.5,0.,-.5], 'Context rows receive no direct supervised gradient'
    try:fn(p,torch.tensor([1]),torch.tensor([1.]),torch.tensor([8]),7)
    except ValueError:pass
    else:raise AssertionError('An immature training label leaked')

def check_messages(fn):
    x={'person':torch.tensor([[1.],[2.]]),'event':torch.tensor([[3.],[5.]])}
    edges={('event','to','person'):torch.tensor([[0,1],[0,0]]),('person','rev','event'):torch.tensor([[0,0],[0,1]])}
    out=fn(x,edges)
    assert out['person'].tolist()==[[8.],[0.]] and out['event'].tolist()==[[1.],[1.]], 'Aggregate into typed receivers'

CHECKS=[('key_edges',check_keys),('visible_rows',check_visibility),('seed_positions',check_roots),('seed_loss',check_loss),('typed_messages',check_messages)]
if __name__=='__main__':
    from relkit import exam_l120 as m
    for name,check in CHECKS:check(getattr(m,name))
    report=dict(status='PASS',live_tasks=[name for name,check in CHECKS])
    (Path(__file__).parent/'_check_l120_results.json').write_text(json.dumps(report,indent=2))
    print(report)

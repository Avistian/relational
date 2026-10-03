"""Independent scalar mask oracle and attention/visibility contracts."""
import torch
from relkit.rt_b10 import relational_masks, safe_attention, eligible_rows

def fixture():
    # Three tables: task(0), customer(1), order(2). Row IDs are global.
    node=torch.tensor([[100,100,10,10,20,20,101,101,11,-1]])
    b=dict(node_idxs=node, f2p_nbr_idxs=torch.tensor([[[10],[10],[-2],[-2],[10],[10],[10],[10],[-2],[-2]]]),
           table_name_idxs=torch.tensor([[0,0,1,1,2,2,0,0,1,-1]]),
           col_name_idxs=torch.tensor([[0,1,0,2,0,1,0,1,0,-1]]),is_padding=node<0)
    return b

def oracle(b):
    n=b['node_idxs'].shape[1];out={k:torch.zeros((1,n,n),dtype=torch.bool) for k in ('col','feat','nbr','full')}
    for i in range(n):
      for j in range(n):
        if b['is_padding'][0,i] or b['is_padding'][0,j]:continue
        a=int(b['node_idxs'][0,i]);c=int(b['node_idxs'][0,j])
        out['col'][0,i,j]=bool(b['table_name_idxs'][0,i]==b['table_name_idxs'][0,j] and b['col_name_idxs'][0,i]==b['col_name_idxs'][0,j])
        out['feat'][0,i,j]=(a==c or c in b['f2p_nbr_idxs'][0,i].tolist())
        out['nbr'][0,i,j]=a in b['f2p_nbr_idxs'][0,j].tolist()
        out['full'][0,i,j]=True
    return out

def checks():
    b=fixture();actual=relational_masks(b)
    for key,expected in oracle(b).items():assert torch.equal(actual[key],expected),key+' has wrong cell access'
    assert actual['feat'][0,0,2] and not actual['feat'][0,2,0], 'FK direction reversed'
    assert actual['nbr'][0,2,4] and not actual['nbr'][0,4,2], 'Neighbor direction reversed'
    assert not actual['col'][0,0,2], 'Same column ID in different tables is not one column'
    q=torch.zeros(1,1,3,2,dtype=torch.float64)
    v=torch.tensor([[[[2.,0.],[6.,0.],[30.,0.]]]],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[[True,True,False],[False,False,False],[True,False,False]]])
    out=safe_attention(q,q,v,mask)
    assert torch.equal(out,torch.tensor([[[[4.,0.],[0.,0.],[2.,0.]]]],dtype=torch.float64)), 'Empty neighbor sets must produce zero, not NaN'
    out.sum().backward();assert torch.isfinite(v.grad).all() and not v.grad[0,0,2].any(), 'Forbidden value affected gradient'
    # Day units; unknown arrival is rejected. Historical labels require completed horizon.
    times=torch.tensor([7.,11.,8.,6.,7.]);arrivals=torch.tensor([7.,9.,8.,float('nan'),12.]);labels=torch.tensor([False,False,True,False,False])
    keep=eligible_rows(times,arrivals,10.,labels,3.)
    assert keep.tolist()==[True,False,False,False,False], 'Time, arrival, or label horizon missing'
    return {'status':'PASS','mask_pairs':400,'contracts':['directed FK masks','table-qualified columns','empty neighborhoods','forbidden gradient','event/arrival/horizon visibility']}
if __name__=='__main__':print(checks())

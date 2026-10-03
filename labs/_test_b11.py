"""Independent learner checks; every operation is used downstream."""
import torch
from relkit.baselines_b11 import eligible, explicit_attention, first_validation_min

def check_eligible(fn):
    e=torch.tensor([7.,8.,12.,10.]);a=torch.tensor([7.,11.,9.,10.])
    assert fn(e,a,10.).tolist()==[True,False,False,True]
    assert not fn(torch.tensor([float('nan')]),torch.tensor([1.]),10.).any()

def check_attention(fn):
    q=torch.zeros(1,1,2,2,dtype=torch.float64)
    k=q.clone();v=torch.tensor([[[[2.,4.],[6.,8.]]]],dtype=torch.float64)
    mask=torch.tensor([[True,False],[False,False]])
    out=fn(q,k,v,attn_mask=mask)
    torch.testing.assert_close(out,torch.tensor([[[[2.,4.],[0.,0.]]]],dtype=torch.float64))
    v[...,1,:]=9999;torch.testing.assert_close(fn(q,k,v,attn_mask=mask),out)
    torch.testing.assert_close(fn(q,k,v),v.mean(-2,keepdim=True).expand_as(q))

def check_selection(fn):
    assert fn([3.,1.,1.,2.])==1
    for bad in [[],[float('nan')],[float('inf')]]:
        try:fn(bad)
        except ValueError:pass
        else:raise AssertionError('Reject empty/nonfinite validation history')

def run():
    check_eligible(eligible);check_attention(explicit_attention);check_selection(first_validation_min)
    rejected=0
    for check,wrong in [(check_eligible,lambda e,a,c:e<=c),(check_attention,lambda q,k,v,**kw:v.mean(-2,keepdim=True).expand_as(q)),(check_selection,lambda s:len(s)-1-min(range(len(s)),key=lambda i:s[::-1][i]))]:
        try:check(wrong)
        except AssertionError:rejected+=1
    assert rejected==3
    print('PASS: three live contracts; three wrong implementations rejected')

if __name__=='__main__':run()

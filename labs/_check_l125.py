"""Independent hand oracles and adversarial contracts, reused by notebook TODOs."""
import json
from pathlib import Path
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data.stats import StatType
from relkit.frame_l125 import numeric_tokens,categorical_indices,align_rows,fit_and_convert

def check_numeric(fn=numeric_tokens):
    raw=torch.tensor([[30.,float('nan')],[10.,4.]])
    mean=torch.tensor([20.,4.]);scale=torch.tensor([10.,2.])
    w=torch.tensor([[2.,-1.],[3.,2.]],requires_grad=True);b=torch.tensor([[.5,.5],[1.,-1.]],requires_grad=True)
    out=fn(raw,mean,scale,w,b)
    torch.testing.assert_close(out,torch.tensor([[[2.5,-.5],[1.,-1.]],[[-1.5,1.5],[1.,-1.]]]))
    out.sum().backward();assert torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()
    assert torch.isnan(raw[0,1]),'Do not mutate raw data'
    return 'PASS'

def check_categories(fn=categorical_indices):
    raw=torch.tensor([[0,0],[1,2],[-1,1],[0,-1]])
    torch.testing.assert_close(fn(raw,[2,3]),torch.tensor([[1,3],[2,5],[0,4],[1,0]]))
    assert torch.equal(raw,torch.tensor([[0,0],[1,2],[-1,1],[0,-1]]))
    try:fn(torch.tensor([[2,0]]),[2,3])
    except ValueError:pass
    else:raise AssertionError('Reject category IDs outside fitted vocabulary')
    return 'PASS'

def check_alignment(fn=align_rows):
    z=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)
    y=fn([40,7,90],z,[90,40,90]);torch.testing.assert_close(y,torch.tensor([[5.,6.],[1.,2.],[5.,6.]]))
    y.sum().backward();torch.testing.assert_close(z.grad,torch.tensor([[1.,1.],[0.,0.],[2.,2.]]))
    for ids,requested in [([40,40,90],[40]),([40,7,90],[999])]:
        try:fn(ids,z,requested)
        except ValueError:pass
        else:raise AssertionError('Reject duplicate source IDs / missing requests')
    return 'PASS'

def check_fit(fn=fit_and_convert):
    tr=pd.DataFrame({'x':[10.,20.,30.],'c':['a','b','a']});q=pd.DataFrame({'x':[1000.,None],'c':['future',None]})
    ds,tf=fn(tr,q,{'x':stype.numerical,'c':stype.categorical})
    assert ds.col_stats['x'][StatType.MEAN]==20
    assert tf.feat_dict[stype.categorical].tolist()==[[-1],[-1]]
    q.x=[1e9,-1e9];other,_=fn(tr,q,{'x':stype.numerical,'c':stype.categorical})
    assert ds.col_stats==other.col_stats,'Query values changed fitted state'
    return 'PASS'

if __name__=='__main__':
    outcomes={}
    for name,fn in [('numeric',check_numeric),('categories',check_categories),('alignment',check_alignment),('fit',check_fit)]:
        try:outcomes[name]=fn()
        except Exception as e:outcomes[name]=type(e).__name__+': '+str(e)
    report={'status':'PASS' if all(v=='PASS' for v in outcomes.values()) else 'FAIL','checks':outcomes}
    print(json.dumps(report,indent=2));(Path(__file__).parent/'_check_l125_results.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='PASS':raise SystemExit(1)

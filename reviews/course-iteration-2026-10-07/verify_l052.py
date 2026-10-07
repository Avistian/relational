"""Independent numerical oracles for the new key-scale explanation."""
from pathlib import Path
import sys,json,math
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'labs'))
import torch
from relkit.tabr import aggregate_context,context_value,select_neighbors
q=torch.tensor([[1.,1.]],dtype=torch.float64);c=torch.tensor([[1.,0.],[3.,1.],[0.,0.]],dtype=torch.float64);allowed=torch.ones(1,3,dtype=torch.bool)
idx=select_neighbors(q,c,2,allowed);assert idx.tolist()==[[0,2]]
v=torch.tensor([[[3.],[4.]]],dtype=torch.float64);drop=torch.nn.Identity();rows=[]
for scale,gap in [(1,1),(2,4)]:
 assert torch.equal(idx,select_neighbors(scale*q,scale*c,2,allowed))
 actual=aggregate_context(scale*q,scale*c[idx],v,drop).item();expected=3+1/(1+math.exp(gap));assert abs(actual-expected)<1e-12
 rows.append(dict(key_scale=scale,context=actual))
shift=torch.tensor([7.,-3.],dtype=torch.float64)
assert torch.equal(idx,select_neighbors(q+shift,c+shift,2,allowed))
torch.testing.assert_close(context_value(q,c[idx],v,drop),context_value(q+shift,(c+shift)[idx],v,drop))
# The direction correction changes even when neighbor order remains fixed.
a=context_value(q,c[idx],torch.zeros(1,2,2),drop);b=context_value(2*q,2*c[idx],torch.zeros(1,2,2),drop);torch.testing.assert_close(b,2*a)
result=dict(status='PASS',traces=rows,checks=['selected IDs under scaling','independent logistic-weight arithmetic','shared translation','directed correction scales'],scope='Exact synthetic mechanism; no benchmark training')
Path(__file__).with_name('l052-check.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

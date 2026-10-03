"""Numerical equation/output/gradient check against pinned QASSMax source; no training."""
import importlib.util,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('b04_upstream_ssmax',P/'sources/b04/upstream/src/tabicl/_model/ssmax.py');up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
torch.set_num_threads(1);torch.manual_seed(404)
m=up.QASSMaxMLP(2,4,elementwise=True).double()
# Nonzero gate weights exercise the query-aware term rather than its zero initialization.
with torch.no_grad():
    m.query_mlp[-1].weight.normal_(0,.1);m.query_mlp[-1].bias.normal_(0,.1)
rows=[]
for n in [2,32,384,15001]:
    q=torch.randn(1,2,3,4,dtype=torch.float64,requires_grad=True)
    actual=m(q,n)
    logn=torch.tensor([[__import__('math').log(n)]],dtype=q.dtype)
    base=m.base_mlp(logn).reshape(1,2,1,4)
    expected=q*base*(1+torch.tanh(m.query_mlp(q)))
    g1=torch.autograd.grad(actual.square().sum(),q,retain_graph=True)[0]
    g2=torch.autograd.grad(expected.square().sum(),q)[0]
    delta=float((actual-expected).detach().abs().max());grad=float((g1-g2).abs().max())
    assert delta<1e-12 and grad<1e-10
    rows.append(dict(support_n=n,output_max_abs=delta,gradient_max_abs=grad))
r=dict(status='PASS',scope='Random-weight query-scaling equation vs pinned source; not a trained-model ablation or Figure 3 reproduction',rows=rows)
(P/'evidence/b04/mechanism-parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

"""Independent mechanism checks; live notebook task checks are extracted by the builder."""
import json
from pathlib import Path
import torch

def check_extract(fn):
    # c1 -> customer -> country <- other customer <- c2.
    # Phase 2 must NOT re-enter phase 1 when it discovers the country.
    edges=[(1,0),(0,2),(3,2),(4,3),(5,1),(1,0)]
    nodes,induced=fn(6,edges,0)
    assert nodes==[0,1,2,5], 'Finish incoming closure, then outgoing closure; do not alternate.'
    assert induced==[(1,0),(0,2),(5,1),(1,0)], 'Preserve parallel induced edges.'
    assert fn(1,[],0)==([0],[]), 'An isolated target remains a datapoint.'
    assert fn(3,[(0,1),(1,0),(2,1)],0)[0]==[0,1,2], 'Cycles must terminate.'

def check_normalize(fn):
    # Reverse edges and self loops already present. Node 0 has degree 3, leaves degree 2.
    edges=torch.tensor([[0,1,0,2,0,1,2],[1,0,2,0,0,1,2]])
    x=torch.tensor([[1.],[2.],[4.]],dtype=torch.float64)
    expected=torch.tensor([[1/3+6/(6**.5)],[1/(6**.5)+1],[1/(6**.5)+2]],dtype=torch.float64)
    assert torch.allclose(fn(x,edges),expected), 'Use symmetric degree normalization, not row means.'
    assert torch.equal(fn(torch.tensor([[3.]]),torch.tensor([[0],[0]])),torch.tensor([[3.]]))

def check_pool(fn):
    values=torch.tensor([[1.,2.],[5.,6.],[9.,10.]])
    gates=torch.tensor([0.,0.,5.]);batch=torch.tensor([0,0,1])
    result=fn(values,gates,batch,2)
    assert torch.allclose(result,torch.tensor([[3.,4.],[9.,10.]])), 'Normalize separately inside each graph.'
    shifted=fn(values,gates+torch.tensor([100.,100.,-100.]),batch,2)
    assert torch.allclose(shifted,result), 'Softmax must be stable and invariant to a per-graph constant.'

if __name__=='__main__':
    from relkit.cvitkovic_l118 import rdb_to_graph,normalized_sum,attention_pool
    check_extract(rdb_to_graph);check_normalize(normalized_sum);check_pool(attention_pool)
    print('PASS: extraction, normalized messages, graph-local readout')

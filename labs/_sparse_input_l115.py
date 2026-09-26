"""Regression: PyG native sparse input adds loops, unlike SparseTensor fill_diag."""
import json
from pathlib import Path
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from relkit.patterns_l115 import normalized_adjacency

def binary_for_reference(edge,n):
    # GCNConv owns loop addition; give it a loop-free binary undirected matrix.
    idx=torch.cat([edge,edge.flip(0)],1)
    idx=idx[:,idx[0]!=idx[1]]
    raw=torch.sparse_coo_tensor(idx,torch.ones(idx.shape[1]),(n,n)).coalesce()
    return torch.sparse_coo_tensor(raw.indices(),torch.ones(raw._nnz()),raw.shape).coalesce().to_sparse_csr()

def check():
    edge=torch.tensor([[0,1],[1,2]]);correct=normalized_adjacency(edge,3)
    c=correct.to_sparse_coo();preloop=torch.sparse_coo_tensor(c.indices(),torch.ones(c._nnz()),c.shape).coalesce().to_sparse_csr()
    wrong,_=gcn_norm(preloop,num_nodes=3)
    assert not torch.allclose(wrong.to_dense(),correct.to_dense()),'Fixture must detect double loops'
    fixed,_=gcn_norm(binary_for_reference(edge,3),num_nodes=3)
    torch.testing.assert_close(fixed.to_dense(),correct.to_dense())
    return {'status':'PASS','double_loop_max_error':float((wrong.to_dense()-correct.to_dense()).abs().max()),'correct_input':'Loop-free binary undirected CSR; original GCNConv adds self-loops','scope':'Replay adapter correction; training graph unchanged'}
if __name__=='__main__':
    r=check();Path(__file__).with_name('_sparse_input_l115_results.json').write_text(json.dumps(r,indent=2));print(r)

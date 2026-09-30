"""Independent matrix-power trace, collision and counterexample."""
import json
from pathlib import Path
import numpy as np,torch
from relkit.pathology_l142 import edge_sum,route_fuse,paired_loss_gap
from _check_l142 import check_edge_sum,check_fuse,check_pair

def run():
    # Coordinate order source, fact, destination, third role.
    adjacency=np.array([[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,1,0,0]],dtype=float)
    transition=np.eye(4)+adjacency
    coefficient=np.linalg.matrix_power(transition,2)[2]
    np.testing.assert_array_equal(coefficient,[1,2,2,1])
    worlds=np.array([[2,1,3,8],[8,1,3,2]])
    ordinary=(transition@transition@worlds.T)[2];composite=worlds[:,0]+worlds[:,1]+worlds[:,2]
    np.testing.assert_array_equal(ordinary,[18,18]);np.testing.assert_array_equal(composite,[6,12])
    # Width-four disjoint role channels preserve all features before any readout.
    encoded=[np.diag(w).sum(0) for w in worlds]
    assert encoded[0][0]!=encoded[1][0]
    # Applying source bias once per fact, even for an empty neighborhood.
    left=torch.nn.Linear(1,1);right=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():left.weight.fill_(2);left.bias.fill_(3);right.weight.fill_(5)
    source=torch.tensor([[7.],[11.]]);fact=torch.tensor([[1.],[2.]])
    edges=torch.tensor([[0,1],[0,0]])
    torch.testing.assert_close(route_fuse(source,fact,edges,left,right),torch.tensor([[44.],[13.]]))
    rejected=[]
    mutants=[('overwrite',check_edge_sum,lambda v,d,n:v.new_zeros((n,)+v.shape[1:]).index_copy_(0,d,v)),('mean_instead_of_sum',check_edge_sum,lambda v,d,n:edge_sum(v,d,n)/2),('drop_fact',check_fuse,lambda s,f,e:edge_sum(s[e[0]],e[1],len(f))),('mix_source_roles',check_fuse,lambda s,f,e:s.sum(0)+f),('positional_pair',check_pair,lambda q,y,ak,a,bk,b:np.abs(np.asarray(b)-y)-np.abs(np.asarray(a)-y)),('reverse_sign',check_pair,lambda *args:-paired_loss_gap(*args))]
    for name,check,fn in mutants:
        try:check(fn)
        except (AssertionError,RuntimeError,ValueError):rejected.append(name)
        else:raise AssertionError(('Mutant survived',name))
    return dict(status='PASS',ordinary_coefficients=coefficient.tolist(),ordinary_worlds=ordinary.tolist(),composite_worlds=composite.tolist(),bias_once_and_empty='PASS',mutants_rejected=rejected,scope='specified linear diagnostic; no impossibility theorem for all GNNs')
if __name__=='__main__':
    r=run();Path(__file__).with_name('_mechanism_l142_results.json').write_text(json.dumps(r,indent=2));print(r)

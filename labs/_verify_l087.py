"""Behavioral oracles for undirected isolation, decoder gradients and tied ranks."""
import json
from pathlib import Path
import numpy as np
import torch
from relkit.link_l087 import canonical_edges, split_edges, sample_non_edges, adjacency, edge_logits, ranking_metrics, heuristic_scores, drnl_subgraph

def verify():
    edges=np.array([[1,0],[0,1],[1,2],[2,3],[3,4],[4,0],[1,3],[2,4],[0,2],[0,0]])
    e=canonical_edges(edges,5)
    assert len(e)==8 and np.all(e[:,0]<e[:,1])
    parts=split_edges(e,5,seed=87,val_fraction=.25,test_fraction=.25)
    keys=[set(map(tuple,p)) for p in parts]
    assert set.union(*keys)==set(map(tuple,e))
    assert all(not keys[i]&keys[j] for i in range(3) for j in range(i))
    a=adjacency(parts[0],5)
    for p in parts[1:]:assert np.all(a[p[:,0],p[:,1]].A1==0) and np.all(a[p[:,1],p[:,0]].A1==0)
    neg=sample_non_edges(e,5,2,np.random.default_rng(87))
    assert len(set(map(tuple,neg)))==2 and not set(map(tuple,neg))&set(map(tuple,e))
    try:sample_non_edges(e,5,3,np.random.default_rng(87))
    except ValueError:pass
    else:raise AssertionError('Dense-graph exhaustion must fail, not hang')
    z=torch.tensor([[1.,2.],[3.,4.],[-1.,2.]],requires_grad=True)
    out=edge_logits(z,torch.tensor([[0,1],[0,2]]));torch.testing.assert_close(out,torch.tensor([11.,3.]))
    out.sum().backward();torch.testing.assert_close(z.grad,torch.tensor([[2.,6.],[1.,2.],[1.,2.]]))
    r=ranking_metrics(np.array([.5,.8]),np.array([[.6,.5,.1],[.7,.2,.1]]),k=1)
    np.testing.assert_allclose(r['ranks'],[2.5,1]);assert r['mrr']==.7 and r['hits_at_k']==.5
    # Independent neighbor-set oracle, including degree-one and isolated nodes.
    a=adjacency(np.array([[0,1],[0,2],[1,3],[2,3],[3,4]]),6)
    pairs=np.array([[0,3],[1,2],[0,5]])
    scores=heuristic_scores(a,pairs)
    np.testing.assert_allclose(scores['CN'],[2,2,0]);np.testing.assert_allclose(scores['RA'],[1,1/2+1/3,0])
    np.testing.assert_allclose(scores['AA'],[2/np.log(2),1/np.log(2)+1/np.log(3),0])
    sub,labels,nodes=drnl_subgraph(a,0,3,hops=1)
    assert sub[0,1]==sub[1,0]==0 and list(labels[:2])==[1,1]
    assert labels[nodes.index(1)]==2 and labels[nodes.index(2)]==2 and labels[nodes.index(4)]==0
    print('PASS: split isolation, negative exhaustion, decoder outputs/gradients, tied ranks, CN/AA/RA, DRNL')
    return {'status':'PASS','checks':['undirected_split_isolation','negative_exhaustion','decoder_outputs_gradients','average_tied_ranks','independent_heuristics','DRNL_target_removal']}

if __name__=='__main__':
    Path(__file__).with_name('_verify_l087_results.json').write_text(json.dumps(verify(),indent=2)+'\n')

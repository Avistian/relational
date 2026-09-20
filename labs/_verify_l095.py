"""Behavioral checks: graph identity, information boundaries, walk and ranking."""
import json
from pathlib import Path
import numpy as np
from relkit.bipartite_l095 import build_graph, assert_boundary, walk_scores, rank_metrics, split_inner

def rejects(fn):
    try: fn()
    except (ValueError,AssertionError): return
    raise AssertionError('Corruption was accepted')

def verify():
    # user:0 and item:0 are different entities; isolated IDs remain represented.
    rows=np.array([[0,0,5,1],[0,1,4,2],[1,1,5,3],[1,2,4,4],[2,0,1,5]])
    g=build_graph(rows,4,4)
    assert g['user'].num_nodes==4 and g['item'].num_nodes==4
    np.testing.assert_array_equal(g['item','rev_likes','user'].edge_index, g['user','likes','item'].edge_index.flip(0))
    held=np.array([[0,2,5,6]])
    assert_boundary(g,held)
    bad=g.clone();bad['item','rev_likes','user'].edge_index[0,0]=2
    rejects(lambda:assert_boundary(bad,held))
    rejects(lambda:build_graph(np.vstack([rows,rows[0]]),4,4))
    rejects(lambda:build_graph(np.array([[4,0,5,1]]),4,4))
    # Enumerate paths independently, rather than repeat the matrix product.
    s,p=walk_scores(g);oracle=np.zeros((4,4));neighbors={0:[0,1],1:[1,2],2:[],3:[]};back={0:[0],1:[0,1],2:[1],3:[]}
    for u in range(2):
        for i in neighbors[u]:
            for v in back[i]:
                for j in neighbors[v]:oracle[u,j]+=1/len(neighbors[u])/len(back[i])/len(neighbors[v])
    oracle[2:]=p
    np.testing.assert_allclose(s,oracle,atol=1e-14)
    np.testing.assert_allclose(s[0],[.375,.5,.125,0])
    assert np.isfinite(s).all();np.testing.assert_allclose(s.sum(1),1)
    # Highest score is an observed dislike: exclude it just like observed likes.
    train=np.array([[0,0,1,1]]);test=np.array([[0,1,5,2],[0,2,5,3],[1,2,1,3]])
    scores=np.array([[100,3,2,1],[1,2,3,4.]])
    m,detail=rank_metrics(scores,train,test,k=1)
    assert m['recall']==.5 and m['ndcg']==1 and m['users']==1 and m['excluded_no_relevant']==1
    assert detail[0]['top_items']==[1]
    # Deterministic tie rule and zero eligible-user reporting.
    assert rank_metrics(np.ones((2,4)),train,test,k=1)[1][0]['top_items']==[1]
    rejects(lambda:rank_metrics(scores,train,np.array([[0,0,5,2]])))
    empty,_=rank_metrics(scores,train,np.array([[1,2,1,3]]));assert empty['ndcg'] is None
    a,b=split_inner(rows,seed=95);assert len(a)+len(b)==len(rows)
    assert not set(map(tuple,a[:,:2]))&set(map(tuple,b[:,:2]))
    return {'status':'PASS','checks':['typed IDs and isolated nodes','reverse-edge corruption rejected','duplicate and out-of-range IDs rejected','independent path enumeration','cold-user fallback','mask all observed ratings','user-macro ranking and denominator','stable ties','empty target report','disjoint inner split']}
if __name__=='__main__':
    result=verify();Path(__file__).with_name('_verify_l095_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

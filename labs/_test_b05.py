"""Behavioral tests with hand-computed answers and adversarial target changes."""
import numpy as np
from relkit.retrieval_b05 import feature_view,neighbors,episode,overlap_status

def check_feature_view(fn):
    t=np.array([[0.,99.,3.],[1.,12.,4.]])
    x,y=fn(t,1);np.testing.assert_array_equal(x,[[0,3],[1,4]]);np.testing.assert_array_equal(y,[99,12])
    x[0,0]=88;assert t[0,0]==0,'Views must not mutate source'
    for target in [-1,3]:
        try:fn(t,target)
        except ValueError:pass
        else:raise AssertionError('Invalid target accepted')

def check_neighbors(fn):
    s=np.array([[0.,0.],[2.,0.],[4.,0.]])
    assert fn(s,[[1.,9.]],[8,2,5],2).tolist()==[[2,8]],'Tie must use row identity'
    a=fn(s,[[1,9],[100,0]],[8,2,5],2);b=fn(s,[[1,9]],[8,2,5],2)
    np.testing.assert_array_equal(a[:1],b)
    for ids,k in [([1,1,2],2),([1,2,3],4),([1,2,3],0)]:
        try:fn(s,[[1,0]],ids,k)
        except ValueError:pass
        else:raise AssertionError('Invalid retrieval contract accepted')

def check_overlap(fn):
    base=dict(dataset_id=1,name='first',sha256='a',family='family-a')
    assert fn(base,dict(base))=='OVERLAP'
    assert fn(base,dict(dataset_id=2,name='renamed',sha256='a',family='unknown'))=='OVERLAP'
    assert fn(base,dict(dataset_id=2,name='renamed',sha256='b',family='family-a'))=='RELATED'
    assert fn(base,dict(dataset_id=2,name='other',sha256='b',family='family-b'))=='NOT_ESTABLISHED'

def run_tests():
    check_feature_view(feature_view);check_neighbors(neighbors);check_overlap(overlap_status)
    t=np.array([[0,0],[.1,100],[.2,0],[.3,100],[1,0],[2,100]],float)
    a=episode(t,1,0,4,2,5);t[:,1]=[900,1,2,3,4,5];b=episode(t,1,0,4,2,5)
    assert a['selected_ids']==b['selected_ids']
    assert set(a['support_ids']).isdisjoint(a['query_ids'])
    assert len(set(a['support_ids']+a['query_ids']))==4
    # Changing labels changes episode outputs but not selected row identities.
    assert a['query_y']!=b['query_y']
    for args in [(1,0,7,2,5),(1,0,4,4,5)]:
        try:episode(t,*args)
        except ValueError:pass
        else:raise AssertionError('Invalid episode accepted')
    return {'status':'PASS','checks':['target removal','source immutability','support-only scaling','stable ties','query batch independence','invalid contracts','target intervention','disjoint identities','overlap boundaries']}
if __name__=='__main__':print(run_tests())

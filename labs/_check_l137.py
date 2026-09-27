"""Behavior checks for silent alignment, unsupported slices and repeated-driver errors."""
import numpy as np
from relkit.error_reg_l137 import paired_errors,nominate_slice,cluster_interval

def rejects(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid contract accepted')

def check_pair(fn):
    keys=[(1,10),(1,20),(2,10)]
    out=fn(keys,[1,2,3],keys[::-1],[5,3,1],keys,[2,2,4])
    np.testing.assert_allclose(out,[-1,1,1])
    rejects(lambda:fn(keys,[1,2,3],keys[:2],[1,2],keys,[2,2,4]))
    rejects(lambda:fn(keys,[1,2,3],[keys[0]]*3,[1,2,3],keys,[2,2,4]))
    rejects(lambda:fn(keys,[1,2,3],keys,[1,float('nan'),3],keys,[2,2,4]))
    rejects(lambda:fn(keys,[1,2,3],[(1,10),(1,21),(2,10)],[1,2,3],keys,[2,2,4]))
    rejects(lambda:fn(keys,[1,2,3],keys,[1,2,3],keys,[2,2]))

def check_nominate(fn):
    d=np.array([4.,4.,-2.,-2.]);ids=np.array([1,2,3,4])
    masks={'a':np.array([1,1,0,0],bool),'b':np.ones(4,bool),'tiny':np.array([1,0,0,0],bool)}
    r=fn(d,ids,masks,min_rows=2,min_entities=2)
    assert r['selected']=='a' and not r['slices']['tiny']['supported']
    assert fn(-np.ones(4),ids,{'all':np.ones(4,bool)},min_rows=2,min_entities=2)['selected'] is None
    rejects(lambda:fn(d,ids,masks,split='test'))
    assert fn(d,np.ones(4),masks,min_rows=2,min_entities=2)['selected'] is None
    rejects(lambda:fn(d,ids,{'bad':np.ones(3,bool)}))

def check_cluster(fn):
    r=fn([2,2,2,2],[1,1,2,2],draws=100,seed=9)
    assert r['mean']==2 and r['low']==2 and r['high']==2 and r['entities']==2
    r=fn([0,0,6],[1,1,2],draws=1000,seed=4)
    assert r['mean']==2 and r['low']==0 and r['high']==6
    # Two draws of driver1 carry four rows; driver2 twice carries two rows.
    assert fn([1],[1])['status']=='INSUFFICIENT_ENTITIES'
    rejects(lambda:fn([1,float('inf')],[1,2]))
    rejects(lambda:fn([1,2],[1]))
    assert fn([0,0,6],[1,1,2],draws=100,seed=4)==fn([0,0,6],[1,1,2],draws=100,seed=4)

if __name__=='__main__':
    check_pair(paired_errors);check_nominate(nominate_slice);check_cluster(cluster_interval)
    print('PASS: keyed pairing, validation-only nomination, driver-cluster resampling')

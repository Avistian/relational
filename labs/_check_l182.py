"""Behavioral contracts; the notebook passes its live learner functions here."""
import numpy as np

def checks(fuse, auc, interaction):
    source=np.array([[1.,2.],[3.,4.]])
    bridge=np.array([[5.,6.],[7.,8.],[9.,10.]])
    args=(source,bridge,np.array([0,1,0]),np.array([0,0,1]),np.array([2.,7.,5.]),np.array([4.,6.]),np.eye(2),2*np.eye(2))
    z,d=fuse(*args)
    np.testing.assert_allclose(z,[[11,14],[19,22]])
    np.testing.assert_array_equal(d,[0,1])
    # Owner cutoff, not batch maximum: event5 for owner0 must be excluded.
    a=list(args);a[4]=np.array([5.,7.,5.]);z,d=fuse(*a)
    np.testing.assert_array_equal(d,[1])
    a=list(args);a[4]=np.array([4.,7.,6.]);z,d=fuse(*a)
    assert z.shape==(0,2) and d.shape==(0,) # Strict before cutoff.
    for index,value in [(2,np.array([0,3,0])),(3,np.array([0,-1,1])),(4,np.array([2.,np.nan,5.]))]:
        a=list(args);a[index]=value
        try:fuse(*a)
        except ValueError:pass
        else:raise AssertionError('Invalid route accepted')
    keys=np.array([[1,10],[1,20],[2,20],[3,20]])
    y=np.array([0,1,0,1]);p=np.array([.2,.8,.8,.9]);order=[3,1,0,2]
    assert auc(keys,y,keys[order],p[order])==.875
    for pk,prob in [(keys[:-1],p[:-1]),(keys[[0,0,2,3]],p),(keys,np.array([.2,.8,np.nan,.9])),(keys,np.array([.2,1.1,.8,.9]))]:
        try:auc(keys,y,pk,prob)
        except ValueError:pass
        else:raise AssertionError('Bad predictions accepted')
    try:auc(keys,np.zeros(4),keys,p)
    except ValueError:pass
    else:raise AssertionError('Single class accepted')
    scores=np.array([[.60,.65,.64,.66],[.61,.66,.65,.67]])
    np.testing.assert_allclose(interaction(scores),[-.03,-.03])
    try:interaction(np.ones((2,3)))
    except ValueError:pass
    else:raise AssertionError('Missing factorial arm accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.composite_l182 import legal_fusion,keyed_auc,factorial_interaction
    print(checks(legal_fusion,keyed_auc,factorial_interaction))

"""Independent hand oracles and adversarial learner-function checks."""
import numpy as np
from relkit.adaptation_b12 import eligible_support,label_attention,intervene

def check_eligibility(fn):
    e=np.array([1.,1.,1.,np.nan,-8.]);a=np.array([2.,12.,2.,2.,-1.]);h=np.array([2.,2.,20.,2.,1.])
    assert np.array_equal(fn(e,a,h,10.),[True,False,False,False,True]), 'Check arrival and completed outcome windows; negative times can be valid'
    assert fn(np.array([8.]),np.array([10.]),np.array([2.]),10.)[0], 'Boundary is inclusive'
    try:fn(e,a,-np.ones(5),10.)
    except ValueError:pass
    else:raise AssertionError('Negative label horizons accepted')

def check_attention(fn):
    q=np.zeros((2,1));k=np.ones((3,1));v=np.array([0.,1.,np.nan]);mask=np.array([[True,True,False],[False,False,False]])
    np.testing.assert_allclose(fn(q,k,v,mask),[.5,.5],atol=1e-12,err_msg='Equal logits average admitted labels; empty support uses declared prior .5')
    q=np.array([[1.]]);k=np.array([[0.],[np.log(3.)]]);v=np.array([0.,1.])
    np.testing.assert_allclose(fn(q,k,v,np.ones((1,2),bool)),[.75],atol=1e-12)
    np.testing.assert_allclose(fn(q,k[::-1],v[::-1],np.ones((1,2),bool)),[.75],atol=1e-12)
    try:fn(q,k,np.array([0.,np.nan]),np.ones((1,2),bool))
    except ValueError:pass
    else:raise AssertionError('Visible unknown label accepted')

def check_intervention(fn):
    y=np.array([0.,1.,1.,0.]);original=y.copy();perm=np.array([1,2,3,0])
    assert np.array_equal(fn(y,'intact',perm),y)
    assert np.array_equal(fn(y,'shuffled',perm),[1.,1.,0.,0.])
    assert np.isnan(fn(y,'hidden',perm)).all()
    assert np.array_equal(y,original),'Intervention mutated original labels'
    for mode,p in [('invalid',perm),('shuffled',np.array([0,0,2,3]))]:
        try:fn(y,mode,p)
        except ValueError:pass
        else:raise AssertionError('Invalid mode or permutation accepted')

if __name__=='__main__':
    for fn,check in [(eligible_support,check_eligibility),(label_attention,check_attention),(intervene,check_intervention)]:check(fn)
    print('PASS: temporal boundary, hidden/empty supports, arithmetic, permutation and input isolation')

"""Analytical, intervention and malformed-input oracles, independent of implementation."""
import numpy as np

def check_functions(attention_readout, fit_missingness, transform_missingness):
    # Scalar hand oracle: logits log(3),0 produce weights .75,.25.
    out,p,h=attention_readout([1.],[[np.log(3.)],[0.]],[[2.],[10.]])
    np.testing.assert_allclose(p,[.75,.25],atol=1e-14)
    np.testing.assert_allclose(out,[4.],atol=1e-14)
    assert abs(h-(-.75*np.log(.75)-.25*np.log(.25))/np.log(2))<1e-14
    _,p2,h2=attention_readout([1.],[[np.log(3.)+10000],[10000]],[[2.],[10.]])
    np.testing.assert_allclose(p2,p,atol=1e-12)
    _,u,hu=attention_readout([0.],[[1.],[8.],[9.]],np.eye(3))
    np.testing.assert_allclose(u,[1/3]*3);assert abs(hu-1)<1e-14
    assert attention_readout([1.],[[2.]],[[3.]])[2]==0
    # Appending distractors lowers anchor mass; log(n) scaling changes that calculation.
    mass=[]
    for n in [2,8,64]:
        k=np.zeros((n,1));k[0]=2
        _,a,_=attention_readout([1.],k,np.eye(n))
        _,b,_=attention_readout([1.],k,np.eye(n),np.log(n))
        assert abs(a[0]-np.exp(2)/(np.exp(2)+n-1))<1e-13
        assert abs(b[0]-n*n/(n*n+n-1))<1e-13
        mass.append(a[0])
    assert mass[0]>mass[1]>mass[2]
    s=np.array([[1.,np.nan,np.nan],[3.,4.,np.nan],[np.nan,8.,np.nan]])
    before=s.copy();mu=fit_missingness(s)
    np.testing.assert_equal(s,before);np.testing.assert_equal(mu,[2.,6.,0.])
    q=np.array([[np.nan,100.,np.nan],[99.,np.nan,1.]])
    z=transform_missingness(q,mu,True)
    np.testing.assert_equal(z,[[2,100,0,1,0,1],[99,6,1,0,1,0]])
    np.testing.assert_equal(mu,[2,6,0]);np.testing.assert_equal(fit_missingness(s),mu)
    # Changing query values cannot change the learned support means.
    q[0,1]=-1e6;assert transform_missingness(q,mu)[1,1]==6
    cases=[lambda:fit_missingness([[np.inf]]),lambda:fit_missingness([]),
           lambda:transform_missingness([[1,2]],[0]),
           lambda:attention_readout([1],[],[]),
           lambda:attention_readout([1],[[float('nan')]],[[1]]),
           lambda:attention_readout([1],[[1]],[[1]],float('inf'))]
    for f in cases:
        try:f()
        except ValueError:pass
        else:raise AssertionError('Malformed input accepted')
    return {'status':'PASS','oracles':'analytical attention, entropy endpoints, distractors, support-only means, empty columns, invalid inputs'}

if __name__=='__main__':
    from relkit.scalable_b04 import attention_readout,fit_missingness,transform_missingness
    print(check_functions(attention_readout,fit_missingness,transform_missingness))

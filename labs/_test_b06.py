"""Behavior checks: mixing boundaries, information isolation and paired identities."""
import numpy as np
from relkit.prior_b06 import choose_prior, support_normalize, paired_effect

def check_prior(fn):
    assert [fn(x,.5) for x in [0,.49,.5,.99]] == ['scm','scm','tree','tree']
    assert fn(0,0)=='tree' and fn(.999,1)=='scm'
    for u,p in [(-.1,.5),(1,.5),(.3,1.1),(float('nan'),.5)]:
        try:fn(u,p)
        except ValueError:pass
        else:raise AssertionError('Reject invalid mixture probabilities/uniform draws')

def check_normalize(fn):
    s=np.array([[1.,7.],[3.,7.]])
    q=np.array([[5.,9.]])
    a,b=fn(s,q)
    np.testing.assert_allclose(a,[[-1,0],[1,0]])
    np.testing.assert_allclose(b,[[3,2e6]])
    c,d=fn(s,q*100)
    np.testing.assert_array_equal(a,c)
    np.testing.assert_array_equal(s,[[1,7],[3,7]])
    assert np.isfinite(d).all()

def check_pair(fn):
    a={'seed0':.4,'seed1':.8};b={'seed1':.3,'seed0':.2}
    np.testing.assert_allclose(fn(a,b),[.2,.5])
    try:fn(a,{'seed0':.1,'seed2':.9})
    except ValueError:pass
    else:raise AssertionError('Mismatched seed identities accepted')

if __name__=='__main__':
    for fn,arg in [(check_prior,choose_prior),(check_normalize,support_normalize),(check_pair,paired_effect)]:fn(arg)
    print('PASS: three learner contracts')

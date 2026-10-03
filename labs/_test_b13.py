"""Hand-oracle contracts: isomorphism, FK selection and regularized prediction."""
import itertools
import numpy as np
from relkit.synthetic_b13 import canonical_schema,parent_mean,fit_ridge

def test_schema():
    chain=[(0,1),(1,2),(2,3)]
    reference=canonical_schema(chain)
    for p in itertools.permutations(range(4)):
        assert canonical_schema([(p[a],p[b]) for a,b in chain])==reference
    assert canonical_schema([(0,1),(0,2),(0,3)])!=reference
    for bad in [[(0,1),(1,0)],[(0,0)],[(0,4)],[(0,1),(0,1)]]:
        try:canonical_schema(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid graph admitted')

def test_parent_mean():
    values=[np.array([2.,6.]),np.array([10.,14.])]
    keys=[np.array([0,1,1]),np.array([1,0,1])]
    np.testing.assert_array_equal(parent_mean(values,keys),[8.,8.,10.])
    try:parent_mean(values,[np.array([2,0,1]),keys[1]])
    except ValueError:pass
    else:raise AssertionError('Dangling FK accepted')

def test_ridge():
    # Standardized x has variance1. With alpha2, coefficient=2/(2+2)=.5.
    model=fit_ridge(np.array([[-1.],[1.]]),np.array([-1.,1.]),2.)
    np.testing.assert_allclose(model['coef'],[.5],atol=1e-14)
    np.testing.assert_allclose(model['mean'],[0.]);np.testing.assert_allclose(model['scale'],[1.])
    const=fit_ridge(np.ones((3,1)),np.array([1.,2.,3.]))
    np.testing.assert_array_equal(const['coef'],[0.]);assert const['intercept']==2.

def checks():
    test_schema();test_parent_mean();test_ridge()
    return {'schema_relabeling_24':'PASS','cycle_duplicate_dangling_rejection':'PASS','parent_read_hand_oracle':'PASS','ridge_hand_oracle':'PASS'}

if __name__=='__main__':print(checks())

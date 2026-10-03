"""Behavioral oracles and malformed-evidence checks for learner functions."""
import numpy as np
from relkit.pfn_b03 import posterior_predictive,align_probabilities,compare_variants

def check_functions(posterior_predictive,align_probabilities,compare_variants):
    # Two latent hypotheses: three successes update [1/2,1/2] to [1/28,27/28].
    p=np.array([.5,.5]);l=np.array([.25**3,.75**3]);q=np.array([.25,.75])
    assert abs(posterior_predictive(p,l,q)-41/56)<1e-12
    assert posterior_predictive(p,np.ones(2),q)==.5
    assert abs(posterior_predictive(p,10*l,q)-41/56)<1e-12
    assert posterior_predictive([1,0],[1,1],[.2,.9])==.2
    # An asymmetric 3-cycle detects the common inverse-permutation mistake.
    x=np.array([[.2,.1,.7],[.6,.3,.1]])
    assert np.allclose(align_probabilities(x,[2,0,1]),[[.7,.2,.1],[.1,.6,.3]])
    assert np.array_equal(x,[[.2,.1,.7],[.6,.3,.1]])
    contract=dict(version='2',variant='base',checkpoint='sha',recipe='four-view',data='blood',splits='official',budget='fixed')
    assert compare_variants(contract,dict(contract))=='MATCHED_CONTRACT'
    for key in contract:
        other=dict(contract);other[key]='different'
        assert compare_variants(contract,other)=='INCOMPARABLE'
        other=dict(contract);other[key]=None
        assert compare_variants(contract,other)=='INCOMPLETE'
    assert compare_variants({}, {})=='INCOMPLETE'
    cases=[(posterior_predictive,([.5,.5],[0,0],[.2,.8])),
           (posterior_predictive,([1,-1],[1,1],[.2,.8])),
           (posterior_predictive,([1],[float('nan')],[.2])),
           (posterior_predictive,([1],[1],[1.1])),
           (align_probabilities,(x,[0,0,2])),
           (align_probabilities,(x,[0,1])),
           (align_probabilities,([[.1,.1,.1]],[0,1,2]))]
    for fn,args in cases:
        try:fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid evidence accepted')
    return dict(status='PASS',posterior='41/56',three_cycle_alignment='PASS',invalid_inputs_rejected=len(cases),contract_fields_checked=len(contract))
if __name__=='__main__':print(check_functions(posterior_predictive,align_probabilities,compare_variants))

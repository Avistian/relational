"""Behavioral oracles for learner work and the paid-run guard."""
import copy
import numpy as np
from relkit.embeddings_b02 import piecewise_linear,member_predictions,greedy_validation,reserve_budget

def check_functions(ple, members, select):
    np.testing.assert_allclose(ple(np.array([-1.,0.,1.,2.,4.,5.]),np.array([0.,2.,4.])),[[-.5,0],[0,0],[.5,0],[1,0],[1,1],[1,1.5]])
    try:ple(np.array([0.]),np.array([0.,0.,1.]))
    except ValueError:pass
    else:raise AssertionError('Duplicate edges accepted')
    x=np.array([[1.,2.],[3.,4.]]);w=np.array([[2.,-1.],[.5,3.]])
    r=np.array([[1.,1.],[-1.,2.]]);s=np.array([[1.,1.],[2.,-1.]]);b=np.array([[.2,.3],[.4,.5]])
    oracle=np.array([[(x[i]*r[k])@w*s[k]+b[k] for k in range(2)] for i in range(2)])
    np.testing.assert_allclose(members(x,w,r,s,b),oracle)
    # Average of two individually imperfect members is exact; selecting on test would differ.
    p=np.array([[0.,2.],[2.,0.],[5.,5.]])
    ids=select(p,np.array([1.,1.]),4)
    assert ids==[0,1],ids
    assert select(np.array([[1.,1.],[1.,1.]]),np.array([1.,1.]),4)==[0]
    try:select(np.array([[np.nan]]),np.array([1.]))
    except ValueError:pass
    else:raise AssertionError('Nonfinite predictions accepted')
    return {'piecewise_extrapolation':'PASS','member_axes':'PASS','validation_selection_ties_stop':'PASS'}

def check_budget():
    base=dict(cap_usd=10,stop_usd=8,overhead_usd=2,reservations=[])
    b=reserve_budget(base,'pilot',600,.0008)
    assert base['reservations']==[] and len(b['reservations'])==1
    for phase,seconds in [('pilot',600),('large',10000),('bad',-1)]:
        try:reserve_budget(b,phase,seconds,.0008)
        except ValueError:pass
        else:raise AssertionError('Unsafe reservation admitted')

if __name__=='__main__':
    print(check_functions(piecewise_linear,member_predictions,greedy_validation));check_budget();print('budget PASS')

"""Learner checks exercise information removal, held-out independence and key alignment."""
import numpy as np
import pandas as pd
from relkit.semantic_b07 import intervene,fit_probe,paired_delta


def check_intervention(fn):
    frame=pd.DataFrame({'name':['a','b'], 'volume':[2.,3.], 'rating':[4.,5.]},index=[8,2])
    original=frame.copy(deep=True)
    full=fn(frame,'meaningful');anon=fn(frame,'anonymous');numeric=fn(frame,'numeric_only')
    pd.testing.assert_frame_equal(full,original)
    assert list(anon.columns)==['6','7','8'],'Use fixed original feature slots'
    np.testing.assert_array_equal(anon.to_numpy(),frame.to_numpy())
    assert list(numeric.columns)==['7','8'],'Do not renumber retained numeric columns'
    np.testing.assert_array_equal(numeric.to_numpy(),frame[['volume','rating']].to_numpy())
    assert list(numeric.index)==[8,2],'Preserve row identities'
    wide=pd.DataFrame({f'field{i}':[1.] for i in range(15)})
    assert list(fn(wide,'anonymous').columns)==[str(i) for i in range(6,19)]+['20','0'],'Only verified cached digit names'
    pd.testing.assert_frame_equal(frame,original)
    try:fn(frame,'typo')
    except ValueError:pass
    else:raise AssertionError('Unknown intervention accepted')


def check_selection(fn):
    x=np.array([[0.],[1.],[2.],[3.],[20.],[40.]])
    y=np.array([0.,1.,2.,3.,0.,0.]);tr=np.array([0,1,2]);va=np.array([3]);te=np.array([4,5])
    a=fn(x,y,tr,va,te)
    yy=y.copy();yy[te]=1e10;xx=x.copy();xx[te]*=100
    b=fn(xx,yy,tr,va,te)
    assert a['alpha']==b['alpha']==1.,'Select alpha using validation only'
    np.testing.assert_allclose(a['mean'],[1.]);np.testing.assert_allclose(a['scale'],[np.sqrt(2/3)])
    np.testing.assert_allclose(a['coefficient'],b['coefficient'])
    np.testing.assert_allclose(a['validation_mse'],b['validation_mse'])
    np.testing.assert_allclose(a['prediction'],[15.25,30.25])
    assert len(a['validation_mse'])==3
    try:fn(x,y,tr,np.array([2,3]),te)
    except ValueError:pass
    else:raise AssertionError('Overlapping train/validation accepted')


def check_pairing(fn):
    a={('table',0):.8,('table',1):.6};b={('table',1):.5,('table',0):.6}
    np.testing.assert_allclose(fn(a,b),[.2,.1],atol=1e-14)
    for left,right in [(a,{('table',0):.6}),({},{}),(a,{('table',0):.6,('wrong',1):.5})]:
        try:fn(left,right)
        except ValueError:pass
        else:raise AssertionError('Missing or different identities accepted')


if __name__=='__main__':
    check_intervention(intervene);check_selection(fit_probe);check_pairing(paired_delta)
    print('PASS: intervention, validation-only selection, paired identities')

"""Independent equation and information-flow checks; reusable notebook feedback."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from relkit.validation_audit_l059 import *
CHECKS={
'choose_validation':'''# CHECK — deterministic ties and malformed score vectors
assert choose_validation([.3,.1,.1])==1, 'Keep the first equal minimum'
for bad in [[],[float('nan')],[.2,float('inf')],[[.1,.2]]]:
    try:choose_validation(bad)
    except ValueError:pass
    else:raise AssertionError('Reject empty, nonfinite or non-vector scores')''',
'loo_residuals':'''# CHECK — compute deleted predictions by actual refitting (independent oracle)
x0=np.array([[0.,0.],[.2,.6],[.9,.3],[1.,.8],[.4,.2]])
y0=np.array([-1.,1.,-1.,1.,1.]); config0=dict(regularization=.3,eta=[2.,.5])
m0=krr_fit(x0,y0,**config0)
brute=[]
for i in range(len(y0)):
    keep=np.arange(len(y0))!=i
    brute.append(y0[i]-predict_krr(krr_fit(x0[keep],y0[keep],**config0),x0[i:i+1])[0])
np.testing.assert_allclose(loo_residuals(m0['alpha'],m0['inverse_diagonal']),brute,atol=1e-10,
  err_msg='Use the diagonal of the augmented inverse; fitted residuals are not deleted residuals')
try:loo_residuals([1.],[0.])
except ValueError:pass
else:raise AssertionError('Zero inverse diagonal must be rejected')''',
'select_krr':'''# CHECK — three candidates, brute-force leave-one-out loss ordering
configs0=[dict(regularization=l,eta=[2.,.5]) for l in [.01,.3,3.]]
oracle=[]
for c in configs0:
    residual=[]
    for i in range(len(y0)):
        keep=np.arange(len(y0))!=i
        residual.append(y0[i]-predict_krr(krr_fit(x0[keep],y0[keep],**c),x0[i:i+1])[0])
    oracle.append(np.mean(np.square(residual)))
model0,index0,loss0=select_krr(x0,y0,configs0)
np.testing.assert_allclose(loss0,oracle,atol=1e-10)
assert index0==int(np.argmin(oracle)), 'Select PRESS, not fitted training error'
np.testing.assert_allclose(predict_krr(model0,x0),predict_krr(krr_fit(x0,y0,**configs0[index0]),x0))''',
'nested_predictions':'''# CHECK — mutate one outer fold's targets; its choice/predictions must not move
xx,yy=mixture(16,590);ff=np.arange(16)%4
p0,tr0=nested_predictions(xx,yy,ff,configs0)
changed=yy.copy();changed[ff==2]*=-1
p1,tr1=nested_predictions(xx,changed,ff,configs0)
np.testing.assert_allclose(p0[ff==2],p1[ff==2],atol=1e-12,
   err_msg='Held-out labels leaked into their own prediction through selection')
a=next(r for r in tr0 if r['fold']==2);b=next(r for r in tr1 if r['fold']==2)
assert a==b, 'Choice and selection loss for the affected fold must stay frozen'
assert sorted(i for r in tr0 for i in r['held_ids'])==list(range(16))
assert all(set(r['fit_ids']).isdisjoint(r['held_ids']) for r in tr0)
for bad in [np.zeros(16,int),np.zeros(15,int),np.arange(16,dtype=float)]:
    try:nested_predictions(xx,yy,bad,configs0)
    except ValueError:pass
    else:raise AssertionError('Invalid fold assignment accepted')''',
'paired_difference':'''# CHECK — a shared shock cancels in paired differences
left=np.array([1.,100.,1000.]);right=left-2.
s=paired_difference(left,right)
assert s['mean']==2 and s['mc_se']==0 and s['t95']==[2.,2.], 'Compute differences before estimating uncertainty'
for a,b in [([1.],[2.]),([1.,2.],[1.]),([1.,np.nan],[1.,2.])]:
    try:paired_difference(a,b)
    except ValueError:pass
    else:raise AssertionError('Invalid repeated observations accepted')'''
}

def check():
    env=globals().copy()
    with threadpool_limits(1):
        for name,code in CHECKS.items():exec(code,env)
        # Equation (3) residual and constant response test expose omitted intercept.
        x,y=mixture(12,7);m=krr_fit(x,y,.1,[.25,2.])
        np.testing.assert_allclose((kernel(x,x,m['eta'])+.1*np.eye(12))@m['alpha']+m['bias'],y,atol=1e-10)
        assert abs(m['alpha'].sum())<1e-10
        flat=krr_fit(x,np.ones(12)*3,.1,[.25,2.])
        np.testing.assert_allclose(predict_krr(flat,np.array([[100.,100.]])),[3.],atol=1e-10)
        assert abs(exact_null_minimum(1,2)-.25)<1e-14
        assert abs(exact_null_minimum(80,1)-.5)<1e-14
        # Historical simulation: live selector remains in path and reproduces old rows.
        import relkit.benchmark_core as historical
        old=historical.choose_validation;calls=[]
        historical.choose_validation=lambda e:(calls.append(len(e)),choose_validation(e))[1]
        evidence=json.loads((Path(__file__).parent/'_verify_l059_results.json').read_text())
        try:
            for row in evidence['rows']:
                actual=historical.null_search(row['candidates'],80,2000,59+row['seed'])
                assert all(actual[k]==row[k] for k in actual)
        finally:historical.choose_validation=old
        assert len(calls)==1000
    result={'status':'PASS','checks':list(CHECKS)+['augmented_system','unpenalized_intercept','exact_binomial_minimum','1000_historical_rows_replayed'],
      'source_parity':'NOT_ESTABLISHED','equation_parity':'Eq3 and LOO verified against independent refits'}
    return result
if __name__=='__main__':
    r=check();(Path(__file__).parent/'_check_l059_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

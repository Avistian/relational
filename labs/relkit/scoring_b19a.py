"""Visible finite-distribution scoring and calibration boundary for B19a.

Finite support CRPS is exact; it is not a sampled or histogram approximation.
"""
import math,itertools

def crps(values, probabilities, y):
    """E|X-y| - 1/2 E|X-X'|, with independent draws from the forecast."""
    if (not values or len(values)!=len(probabilities)
        or not all(math.isfinite(v) for v in [*values,*probabilities,y])
        or any(p<0 for p in probabilities)
        or not math.isclose(sum(probabilities),1,abs_tol=1e-12,rel_tol=0)):
        raise ValueError('Require finite support, nonnegative unit mass and finite outcome')
    accuracy=sum(p*abs(x-y) for x,p in zip(values,probabilities))
    spread=sum(p*q*abs(x-z) for x,p in zip(values,probabilities)
               for z,q in zip(values,probabilities))
    return accuracy-.5*spread

def interval_score(lower, upper, y, alpha):
    """Width + 2/alpha times distance outside a central 1-alpha interval."""
    if not all(math.isfinite(v) for v in [lower,upper,y,alpha]) or lower>upper or not 0<alpha<1:
        raise ValueError('Require ordered finite endpoints, finite outcome and 0<alpha<1')
    return upper-lower+(2/alpha)*max(lower-y,0)+(2/alpha)*max(y-upper,0)

def calibrate(calibration_rows, alpha):
    """Split-conformal absolute residual quantile; predictor was fixed before calibration.

The API accepts only calibration records, never final evaluation labels.
Infinite radius is intentional when ceil((n+1)(1-alpha)) exceeds n.
"""
    if not calibration_rows or not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('Require calibration records and 0<alpha<1')
    ids=[r['id'] for r in calibration_rows]
    if len(set(ids))!=len(ids) or any(r['split']!='calibration' for r in calibration_rows):
        raise ValueError('Duplicate identity or non-calibration split')
    if not all(math.isfinite(r[k]) for r in calibration_rows for k in ['y','mean']):
        raise ValueError('Require finite targets and fixed predictions')
    residuals=sorted(abs(r['y']-r['mean']) for r in calibration_rows)
    n=len(residuals);k=math.ceil((n+1)*(1-alpha))
    return dict(n=n,k=k,radius=float(residuals[k-1]) if k<=n else float('inf'),ids=ids)

def quantile(values, probabilities, q):
    """Left generalized inverse; use stated endpoints for q=0 and q=1."""
    crps(values,probabilities,0)
    if not math.isfinite(q) or not 0<=q<=1:raise ValueError('Invalid quantile')
    mass=0
    for value,p in sorted(zip(values,probabilities)):
        mass+=p
        if mass>=q:return value
    return max(values)

def run_experiment():
    truth=[(-2,.25),(0,.5),(2,.25)]
    forecasts={'narrow':([-1,1],[.5,.5]),'calibrated':([-2,0,2],[.25,.5,.25]),'wide':([-4,4],[.5,.5])}
    rows=[];summary=[]
    for name,(values,probabilities) in forecasts.items():
        mean=sum(x*p for x,p in zip(values,probabilities));lo=quantile(values,probabilities,.25);hi=quantile(values,probabilities,.75)
        local=[]
        for y,w in truth:
            r=dict(forecast=name,y=y,weight=w,mean=mean,lower=lo,upper=hi,squared_error=(y-mean)**2,crps=crps(values,probabilities,y),interval_score=interval_score(lo,hi,y,.5),covered=lo<=y<=hi,width=hi-lo)
            local.append(r);rows.append(r)
        summary.append(dict(forecast=name,mean=mean,rmse=math.sqrt(sum(r['weight']*r['squared_error'] for r in local)),**{k:sum(r['weight']*r[k] for r in local) for k in ['crps','interval_score','covered','width']}))
    grid=[]
    for a in range(5):
        for b in range(5-a):
            p=[a/4,b/4,(4-a-b)/4]
            grid.append(dict(probabilities=p,expected_crps=sum(w*crps([-2,0,2],p,y) for y,w in truth)))
    cal=[dict(id=f'c{i}',split='calibration',y=i,mean=0) for i in range(9)]
    fitted=calibrate(cal,.2);test=[dict(id=f't{i}',split='test',y=y,mean=0) for i,y in enumerate([0,7,8,20])]
    assert not set(fitted['ids'])&{r['id'] for r in test}
    before=fitted.copy();changed=[dict(r,y=r['y']+100) for r in test]
    after=calibrate(cal,.2)
    interventions=[]
    for r in rows:
        v,p=forecasts[r['forecast']]
        for scale,shift in [(1,7),(10,-3)]:
            interventions.append(dict(forecast=r['forecast'],y=r['y'],scale=scale,shift=shift,crps_scaled=crps([scale*x+shift for x in v],p,scale*r['y']+shift),expected=scale*r['crps'],interval_scaled=interval_score(scale*r['lower']+shift,scale*r['upper']+shift,scale*r['y']+shift,.5),interval_expected=scale*r['interval_score']))
    return dict(protocol='B19a-SAME-MEAN',forecasts={k:dict(values=v,probabilities=p) for k,(v,p) in forecasts.items()},truth=truth,rows=rows,summary=summary,propriety_grid=grid,unit_interventions=interventions,calibration=dict(fitted=fitted,test=test,changed_test=changed,radius_unchanged=before==after,coverage=sum(abs(r['y'])<=fitted['radius'] for r in test)/len(test),shifted_coverage=sum(abs(r['y'])<=fitted['radius'] for r in changed)/len(changed)),paper_training='NOT_RUN')

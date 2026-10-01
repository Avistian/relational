"""Regression portfolio contracts: pure NumPy, live in both notebook lanes."""
import math
import statistics
import numpy as np

def keyed_metrics(queries, predictions):
    """Join complete unique (entity,time) sets; bias means prediction minus target."""
    q={(r['entity'],r['time']):float(r['target']) for r in queries}
    p={(r['entity'],r['time']):float(r['prediction']) for r in predictions}
    if not q or len(q)!=len(queries) or len(p)!=len(predictions) or q.keys()!=p.keys():
        raise ValueError('Require identical nonempty unique query keys')
    if not all(math.isfinite(x) for x in list(q.values())+list(p.values())):
        raise ValueError('Nonfinite target or prediction')
    residual=np.array([p[k]-q[k] for k in sorted(q)],dtype=float)
    return dict(n=len(q),mae=float(np.mean(np.abs(residual))),rmse=float(np.sqrt(np.mean(residual**2))),bias=float(np.mean(residual)))

def median_diagnostics(target, prediction, edges):
    """Fixed prediction bins; ties satisfy both median inequalities.

    Bins are [-inf,e0),[e0,e1),...,[elast,inf). Mean residual is y-p.
    This finite-sample diagnostic does not certify population calibration.
    """
    y=np.asarray(target,dtype=float);p=np.asarray(prediction,dtype=float);e=np.asarray(edges,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or e.ndim!=1 or len(y)!=len(p) or not len(y):raise ValueError('Aligned nonempty vectors required')
    if not all(np.isfinite(x).all() for x in [y,p,e]) or np.any(np.diff(e)<=0):raise ValueError('Finite vectors and strictly increasing edges required')
    group=np.searchsorted(e,p,side='right');rows=[]
    for i in range(len(e)+1):
        mask=group==i;n=int(mask.sum())
        row=dict(bin=i,n=n,prediction_mean=None,below=None,equal=None,above=None,median_violation=None,mean_residual=None)
        if n:
            below=float(np.mean(y[mask]<p[mask]));equal=float(np.mean(y[mask]==p[mask]));above=float(np.mean(y[mask]>p[mask]))
            row.update(prediction_mean=float(p[mask].mean()),below=below,equal=equal,above=above,median_violation=max(0.,below-.5,above-.5),mean_residual=float(np.mean(y[mask]-p[mask])))
        rows.append(row)
    return rows

def portfolio_summary(records):
    """All five full fits, sample seed SD, and explicitly bounded evidence claims."""
    if len(records)!=5 or {r['seed'] for r in records}!=set(range(5)):raise ValueError('Exactly seeds0–4 required')
    for r in records:
        if r['epochs']!=10 or r['complete'] is not True:raise ValueError('Incomplete fit')
        if any(not math.isfinite(r[k]) or r[k]<0 for k in ['val_mae','test_mae','test_rmse']):raise ValueError('Invalid metric')
    result={}
    for name,key in [('val','val_mae'),('test','test_mae'),('rmse','test_rmse')]:
        values=[r[key] for r in sorted(records,key=lambda r:r['seed'])]
        result[name]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),values=values)
    result.update(paper_score='CLOSE' if abs(result['test']['mean']-4.022)<=.20 else 'OUTSIDE_TOLERANCE',historical_identity='NOT_ESTABLISHED',feature_arrival_legality='NOT_ESTABLISHED',whole_paper='NOT_RUN',fresh_fe_comparison='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
    return result

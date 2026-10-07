"""Behavior contracts for real learner functions, including adversarial packets."""
import math
from relkit.regression_l152 import keyed_metrics,median_diagnostics,portfolio_summary

def rejects(f,*args):
    try:f(*args)
    except ValueError:return
    raise AssertionError('Invalid evidence accepted')

def check_metrics(f):
    q=[dict(entity=7,time=10,target=2.),dict(entity=7,time=20,target=5.)]
    p=[dict(entity=7,time=20,prediction=3.),dict(entity=7,time=10,prediction=3.)]
    r=f(q,p);assert r['mae']==1.5 and abs(r['rmse']-math.sqrt(2.5))<1e-12 and r['bias']==-.5 and r['n']==2
    assert r==f(list(reversed(q)),p)
    rejects(f,q,p[:1]);rejects(f,q,[p[0],p[0]]);rejects(f,[q[0],q[0]],p)
    rejects(f,q,[p[0],dict(entity=7,time=11,prediction=3.)])
    rejects(f,q,[p[0],dict(entity=7,time=10,prediction=float('nan'))]);rejects(f,[],[])

def check_diagnostics(f):
    r=f([0,1,1,9],[1,1,1,1],[1,2])
    assert [b['n'] for b in r]==[0,4,0]
    b=r[1];assert b['below']==.25 and b['equal']==.5 and b['above']==.25 and b['median_violation']==0
    assert b['mean_residual']==1.75 # mean bias despite empirical median balance
    assert r[0]['median_violation'] is None
    b=f([0,0,0,9],[1]*4,[])[0];assert b['median_violation']==.25
    assert f([1,1,1],[1]*3,[1])[1]['median_violation']==0
    rejects(f,[1],[1],[2,1]);rejects(f,[1],[1],[1,1]);rejects(f,[1,2],[1],[])
    rejects(f,[1],[float('inf')],[]);rejects(f,[1],[1],[float('nan')])

def check_summary(f):
    rows=[dict(seed=s,epochs=10,complete=True,val_mae=3.193,test_mae=4.022,test_rmse=5.) for s in range(5)]
    r=f(rows);assert r['test']['mean']==4.022 and r['test']['sample_sd']==0 and r['paper_score']=='CLOSE'
    assert r['learner']=='PENDING_WRITTEN_DEFENSE' and r['historical_identity']=='NOT_ESTABLISHED'
    rejects(f,rows[:4]);rejects(f,rows[:4]+[rows[0]])
    for field,value in [('epochs',9),('complete',False),('test_mae',float('nan')),('test_rmse',-1)]:
        broken=[dict(x) for x in rows];broken[0][field]=value;rejects(f,broken)
    assert f([dict(x,test_mae=4.3) for x in rows])['paper_score']=='OUTSIDE_TOLERANCE'
    for boundary in [4.022-.20,4.022+.20]:
        assert f([dict(x,test_mae=boundary) for x in rows])['paper_score']=='CLOSE', 'Inclusive tolerance boundary'
    for outside in [4.022-.20-1e-9,4.022+.20+1e-9]:
        assert f([dict(x,test_mae=outside) for x in rows])['paper_score']=='OUTSIDE_TOLERANCE', 'Reject materially outside boundary'


if __name__=='__main__':
    check_metrics(keyed_metrics);check_diagnostics(median_diagnostics);check_summary(portfolio_summary)
    print('PASS key alignment, finite metrics, median ties/empty bins, complete seed evidence')

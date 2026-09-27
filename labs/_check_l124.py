"""Boundary and identity checks; student checks call the supplied function."""
import json
from pathlib import Path
import pandas as pd
from relkit.tasks_l124 import make_labels,validate_task,aligned_mae

def check_labels(fn):
    t=pd.Timestamp('2020-01-01')
    r=pd.DataFrame({'driverId':[0,0,0,0,1,2], 'date':[t,t+pd.Timedelta(days=1),t+pd.Timedelta(days=60),t+pd.Timedelta(days=61),t+pd.Timedelta(days=2),t-pd.Timedelta(days=400)], 'positionOrder':[99,2,6,99,3,9]})
    a=fn(r,[t]);b=fn(r,[t],cohort='past')
    assert a[['driverId','position']].values.tolist()==[[0,4.],[1,3.]]
    assert b[['driverId','position']].values.tolist()==[[0,4.]]
    assert (a['label_end']==t+pd.Timedelta(days=60)).all()
    assert fn(r,[t+pd.Timedelta(days=500)]).empty
    assert len(fn(r.sample(frac=1,random_state=3),[t]))==2
    try:fn(r,[t,t])
    except ValueError:pass
    else:raise AssertionError('Duplicate cutoffs accepted')

def check_validate(fn):
    t=pd.Timestamp('2020-01-01');end=t+pd.Timedelta(days=60)
    rows=pd.DataFrame({'driverId':[0,0],'date':[t,t+pd.Timedelta(days=60)],'position':[4.,5.],'label_end':[end,end+pd.Timedelta(days=60)]})
    assert fn(rows,[0])==2
    assert fn(rows.iloc[:1],[0],fit_time=end)==1
    for bad,ids,fit in [(pd.concat([rows,rows.iloc[:1]]),[0],None),(rows,[1],None),(rows,[0],end),(rows.assign(position=float('nan')),[0],None),(rows.assign(label_end=t),[0],None)]:
        try:fn(bad,ids,fit)
        except ValueError:pass
        else:raise AssertionError('Invalid task accepted')

def check_alignment(fn):
    t=pd.Timestamp('2020-01-01');u=t+pd.Timedelta(days=60)
    rows=pd.DataFrame({'driverId':[0,0],'date':[t,u],'position':[2.,6.]})
    preds=pd.DataFrame({'driverId':[0,0],'date':[u,t],'prediction':[5.,4.]})
    assert fn(rows,preds)==1.5
    for bad in [preds.iloc[:1],pd.concat([preds,preds.iloc[:1]]),preds.assign(prediction=float('nan'))]:
        try:fn(rows,bad)
        except ValueError:pass
        else:raise AssertionError('Bad prediction identity accepted')

if __name__=='__main__':
    for check,fn in [(check_labels,make_labels),(check_validate,validate_task),(check_alignment,aligned_mae)]:check(fn)
    report={'status':'PASS','checks':['open-left closed-right 60-day window','source versus past-only cohort','censoring is not zero','duplicate queries','dangling entities','label maturity','finite labels','shuffled query-key predictions']}
    Path(__file__).with_name('_check_l124_results.json').write_text(json.dumps(report,indent=2));print(report)

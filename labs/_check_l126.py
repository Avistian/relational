"""Behavioral contracts with independent, hand-computed adversarial cases."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from relkit.beta_l126 import schema_audit, engagement_table, align_predictions, average_precision

def check_schema(fn):
    def tab(df,pk=None,fk=None):return SimpleNamespace(df=df,pkey_col=pk,fkey_col_to_pkey_table=fk or {},time_col=None)
    tabs={'users':tab(pd.DataFrame({'id':[10,20,30]}),'id'), 'events':tab(pd.DataFrame({'id':[1,2,3],'user':[20,None,99]}),'id',{'user':'users'})}
    r=fn(tabs)
    assert r['total_rows']==6 and r['tables']['users']['rows']==3
    edge=r['relations'][0]
    assert edge['resolved']==1 and edge['null']==1 and edge['dangling']==1, 'Null and dangling keys have different meanings'
    tabs['users'].df=pd.DataFrame({'id':[10,10]})
    try:fn(tabs)
    except ValueError:pass
    else:raise AssertionError('Duplicate primary keys accepted')

def fixture():
    t=pd.Timestamp('2019-01-01');end=t+pd.Timedelta(days=730)
    users=pd.DataFrame({'Id':[1,2,3,4,-1,5],'CreationDate':[t-pd.Timedelta(days=10)]*5+[t+pd.Timedelta(days=1)]})
    events=pd.DataFrame({'user':[1,1,2,2,3,4,4,-1,5,5,np.nan], 'time':[t,t+pd.Timedelta(days=1),t-pd.Timedelta(days=1),end,t+pd.Timedelta(days=2),t-pd.Timedelta(days=2),end+pd.Timedelta(seconds=1),t,t-pd.Timedelta(days=1),t+pd.Timedelta(days=1),t]})
    return users,events,t

def check_engagement(fn):
    users,events,t=fixture();r=fn(users,events,[t]).sort_values('OwnerUserId')
    assert r.OwnerUserId.tolist()==[1,2,4], 'Eligibility uses existing, previously active real users'
    assert r.contribution.tolist()==[1,1,0], 'Future window is (cutoff, cutoff+730 days]'
    original=r.copy();changed=events[events.time<=t]
    no_future=fn(users,changed,[t]).sort_values('OwnerUserId')
    assert no_future.OwnerUserId.tolist()==original.OwnerUserId.tolist(), 'Future must not change eligibility'
    assert no_future.contribution.tolist()==[0,0,0], 'Cutoff event belongs to history, not target'
    again=fn(users,events.sample(frac=1,random_state=4),[t])
    pd.testing.assert_frame_equal(r.reset_index(drop=True),again.sort_values('OwnerUserId').reset_index(drop=True))

def check_alignment(fn):
    q=pd.DataFrame({'id':[1,2,1],'time':[10,10,20]});p=pd.DataFrame({'id':[1,1,2],'time':[20,10,10],'score':[.8,.2,.4]})
    np.testing.assert_array_equal(fn(q,p,['id','time']),[.2,.4,.8])
    for bad in [p.iloc[:2],pd.concat([p,p.iloc[:1]],ignore_index=True),p.assign(score=[np.nan,.2,.4]),pd.concat([p,p.assign(id=99).iloc[:1]],ignore_index=True)]:
        try:fn(q,bad,['id','time'])
        except ValueError:pass
        else:raise AssertionError('Missing/extra/duplicate/nonfinite predictions accepted')

def check_ap(fn):
    # Descending thresholds .9,.5,.1; recall increments 1/3 each.
    y=np.array([1,0,1,0,1]);s=np.array([.9,.5,.5,.1,.1])
    np.testing.assert_allclose(fn(y,s),(1+2/3+3/5)/3,atol=1e-14)
    np.testing.assert_allclose(fn([1,0,0,1],[.5]*4),.5)
    assert fn([0,0],[.2,.9])==0 and fn([1,1],[.2,.9])==1
    for yy,ss in [([],[]),([0,2],[.1,.2]),([0,1],[.1]),([0,1],[np.inf,.2])]:
        try:fn(yy,ss)
        except ValueError:pass
        else:raise AssertionError('Invalid evaluator inputs accepted')

if __name__=='__main__':
    for fn,check in [(schema_audit,check_schema),(engagement_table,check_engagement),(align_predictions,check_alignment),(average_precision,check_ap)]:
        check(fn);print('PASS',fn.__name__)
    Path(__file__).with_name('_check_l126_results.json').write_text(json.dumps({'status':'PASS','contracts':4},indent=2)+'\n')

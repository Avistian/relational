"""Execute unmodified beta package classes against independent contracts."""
import contextlib,io,json,sqlite3,sys,warnings
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'sources/l126/beta'))
import numpy as np
import pandas as pd
from relbench.data import Table,Database,Dataset
from relbench.tasks.stackex import EngageTask
from relbench.metrics import average_precision as upstream_ap
from relkit.beta_l126 import engagement_table,average_precision,beta_events,beta_cutoffs
from _check_l126 import fixture

def as_db(users,events):
    # A single source is sufficient for boundary fixtures; random cases use all three.
    tables={'users':Table(users,{},'Id','CreationDate')}
    for k,(name,col) in enumerate([('posts','OwnerUserId'),('comments','UserId'),('votes','UserId')]):
        part=events.iloc[k::3].rename(columns={'user':col,'time':'CreationDate'}).copy()
        part['Id']=np.arange(len(part))
        # Preserve original release sentinel filtering assumptions in source tests.
        part=part.loc[part[col]!=-1]
        tables[name]=Table(part,{col:'users'},'Id','CreationDate')
    return Database(tables)

def original_table(db,cutoffs):
    obj=EngageTask.__new__(EngageTask)
    with contextlib.redirect_stderr(io.StringIO()):return obj.make_table(db,pd.Series(cutoffs)).df

def sql_table(users,events,cutoff):
    u=users.copy();e=events.copy();u['created']=u.CreationDate.astype('datetime64[s]').astype('int64');e['at']=e.time.astype('datetime64[s]').astype('int64')
    c=int(cutoff.timestamp());end=int((cutoff+pd.Timedelta(days=730)).timestamp())
    with sqlite3.connect(':memory:') as conn:
        u[['Id','created']].to_sql('users',conn,index=False);e[['user','at']].to_sql('events',conn,index=False)
        rows=conn.execute('SELECT u.Id, EXISTS(SELECT 1 FROM events e WHERE e.user=u.Id AND e.at>? AND e.at<=?) FROM users u WHERE u.Id != -1 AND u.created<=? AND EXISTS(SELECT 1 FROM events e WHERE e.user=u.Id AND e.at<=?) ORDER BY u.Id',(c,end,c,c)).fetchall()
    return rows

def main():
    rng=np.random.default_rng(126);cases=0;ap_cases=0;max_error=0.
    for trial in range(41):
        if trial==0:users,events,t=fixture()
        else:
            t=pd.Timestamp('2019-01-01')
            users=pd.DataFrame({'Id':np.arange(16),'CreationDate':t+pd.to_timedelta(rng.integers(-100,100,16),unit='D')})
            events=pd.DataFrame({'user':rng.integers(0,18,150),'time':t+pd.to_timedelta(rng.integers(-800,900,150),unit='D')})
        db=as_db(users,events);normalized=beta_events(db.table_dict)
        for cutoff in [t,t+pd.Timedelta(days=50)]:
            actual=engagement_table(users,normalized,[cutoff]).sort_values('OwnerUserId').reset_index(drop=True)
            expected=original_table(db,[cutoff]).sort_values('OwnerUserId').reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,expected[actual.columns],check_dtype=False)
            assert list(actual[['OwnerUserId','contribution']].itertuples(index=False,name=None))==sql_table(users,normalized,cutoff)
            cases+=1
    for n in [2,5,10,50,100]:
        for _ in range(40):
            y=rng.integers(0,2,n);s=rng.integers(-3,4,n)/3
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');expected=upstream_ap(y,s)
            actual=average_precision(y,s);max_error=max(max_error,abs(actual-expected));assert abs(actual-expected)<1e-12
            ap_cases+=1
    # Original dataset cap, split construction, and hidden test target with real package objects.
    users,events,t=fixture()
    users.loc[users.Id==1,"CreationDate"]=t-pd.Timedelta(days=3000)
    events=pd.concat([events,pd.DataFrame({"user":[1],"time":[t-pd.Timedelta(days=2000)]})],ignore_index=True)
    db=as_db(users,events)
    dataset=Dataset(db,pd.Timestamp('2019-01-01'),pd.Timestamp('2021-01-01'),[EngageTask])
    task=EngageTask(dataset,process=True)
    with contextlib.redirect_stderr(io.StringIO()):
        masked=task.test_table;full=task._full_test_table;train=task.train_table
    assert 'contribution' not in masked.df and 'contribution' in full.df
    assert dataset.db.max_timestamp<=dataset.test_timestamp
    expected_times=beta_cutoffs(dataset.db.min_timestamp)['train']
    assert set(train.df.timestamp)<=set(expected_times)
    # Rank-only predictions are a fixture metric, not a fitted model score.
    if len(full.df):
        pred=np.linspace(.1,.9,len(full.df))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');assert abs(task.evaluate(pred,metrics=[upstream_ap])['average_precision']-average_precision(full.df.contribution,pred))<1e-12
    r={'status':'PASS','original_beta_revision':'0433616ee94003fb15a4ac4d633e499d0f129077','task_cases':cases,'independent_sql_cases':cases,'ap_cases':ap_cases,'maximum_ap_error':max_error,'original_dataset_mask_evaluator':'PASS','scope':'Synthetic source/API parity only; historical data NOT_RUN'}
    (P/'_source_check_l126_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()

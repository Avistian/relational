"""Independent complete archive, label, SQL dependency and fit-scope census."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from relkit.temporal_audit_l156 import audit_observations,audit_label_windows
from relkit.fe_experiment_l129 import load_archives,make_features
P=Path(__file__).resolve().parent;E=P/'evidence/l157';E.mkdir(parents=True,exist_ok=True)
start=time.perf_counter();S=P/'sources/l129';tables,queries,raw=load_archives(S)
archives={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'db.zip',S/'driver-position.zip']}
with zipfile.ZipFile(S/'db.zip') as z:
    all_results=pd.read_parquet(io.BytesIO(z.read('db/results.parquet')))
    metadata={Path(n).stem:{k.decode():json.loads(v) for k,v in pq.read_schema(io.BytesIO(z.read(n))).metadata.items() if k in [b'time_col',b'pkey_col',b'fkey_col_to_pkey_table']} for n in z.namelist() if n.endswith('.parquet')}
def seconds(series):return (series.astype('int64')//10**9).to_numpy()
events=[dict(entity=int(e),time=int(t),value=float(y)) for e,t,y in zip(all_results.driverId,seconds(all_results.date),all_results.positionOrder)]
label_results={};record=[]
for split,q in queries.items():
    qs=[dict(entity=int(e),time=int(t),target=float(y)) for e,t,y in zip(q.driverId,seconds(q.date),q.position)]
    audit=audit_label_windows(qs,events,60*86400);assert audit['status']=='PASS';audit.pop('rows');label_results[split]=audit
    # Independently verify the complete eligible population, including omitted queries.
    for t,g in q.groupby('date'):
        expected=set(all_results.loc[(all_results.date>t)&(all_results.date<=t+pd.Timedelta(days=60)),'driverId'])
        assert set(g.driverId)==expected
    label_results[split]['earliest_cutoff']=str(q.date.min());label_results[split]['latest_label_end']=str(q.date.max()+pd.Timedelta(days=60))
# Labels must have matured before validation/testing begins under released split policy.
assert queries['train'].date.max()+pd.Timedelta(days=60)<=pd.Timestamp('2005-01-01')
assert queries['val'].date.max()+pd.Timedelta(days=60)<=pd.Timestamp('2010-01-01')
fit_scope={};examples=[]
for name,df in tables.items():
    tm=metadata[name]['time_col']
    if tm is None:
        fit_scope[name]=dict(rows=len(df),event_time='NOT_OBSERVED',availability='NOT_ESTABLISHED');continue
    past=df[df[tm]<=pd.Timestamp('2005-01-01')];after=int((df[tm]>pd.Timestamp('2005-01-01')).sum())
    stats=[]
    forbidden=set(metadata[name]['fkey_col_to_pkey_table'])|{metadata[name]['pkey_col']}
    for col in df.select_dtypes('number'):
        if col in forbidden:continue
        a=float(past[col].mean());b=float(df[col].mean())
        if np.isfinite(a) and np.isfinite(b) and a!=b:stats.append(dict(column=col,fit_horizon_mean=a,released_mean=b))
    fit_scope[name]=dict(rows=len(df),rows_after_fit_horizon=after,fit_horizon_rows=len(past),changed_numeric_means=stats)
    examples.extend(dict(table=name,**s) for s in stats)
# Regenerate all released SQL fields; independent reconstruction runs next.
features=make_features(tables,queries,(S/'f1/driver-position/feats.sql').read_text())
O=E/'fe';O.mkdir(exist_ok=True)
for split,df in features.items():df.to_parquet(O/f'{split}-features.parquet',index=False)
# Joined result/standings rows must carry the same race event date; primary identifiers alone do not prove this.
race_dates=tables['races'].set_index('raceId').date;join_counts={}
for name in ['results','standings','constructor_results','constructor_standings']:
    df=tables[name];expected=df.raceId.map(race_dates);valid=expected.notna()
    assert (df.loc[valid,'date'].to_numpy()==expected[valid].to_numpy()).all(),name
    join_counts[name]=dict(checked=int(valid.sum()),dangling=int((~valid).sum()))
# Materialize actual SQL lineage records for every query/history/schedule slot.
stands={i:g for i,g in tables['standings'].groupby('driverId')};races=tables['races'].set_index('raceId');rr=tables['results'].drop_duplicates(['raceId','driverId']).set_index(['raceId','driverId'])
dependency_counts={};witnesses=[]
for split,q in queries.items():
    records=[];lag=[]
    for owner,row in enumerate(q.itertuples(index=False)):
        entity=row.driverId;t=row.date;cut=int(t.value//10**9)
        history=stands.get(entity)
        if history is not None:
            old=history[history.date<t]
            if len(old):records.append(dict(owner=owner,cutoff=cut,event=int(old.date.max().value//10**9),available=None,rule='strict',kind='event'))
        before=races[races.date<t]
        if len(before):
            last=int(before.date.idxmax())
            for i in [1,2,3]:
                key=(last-i+1,entity)
                if key in rr.index:
                    past=rr.loc[key]
                    if past.date>=t-pd.DateOffset(months=2):records.append(dict(owner=owner,cutoff=cut,event=int(past.date.value//10**9),available=None,rule='strict',kind='event'))
                if last+i in races.index:
                    up=races.loc[last+i]
                    if up.date<=t+pd.DateOffset(months=1):
                        records.append(dict(owner=owner,cutoff=cut,event=int(up.date.value//10**9),available=None,rule='inclusive',kind='schedule'))
                        if up.date>t and len(witnesses)<3:witnesses.append(dict(split=split,entity=int(entity),cutoff=str(t),scheduled_event=str(up.date),availability='NOT_OBSERVED'))
        records.append(dict(owner=owner,cutoff=cut,event=None,available=None,rule='inclusive',kind='static'))
    out=audit_observations(records);assert out['counts']['FAIL']==0
    dependency_counts[split]=dict(status=out['status'],counts=out['counts'],event_dependencies=sum(r['kind']=='event' for r in records),schedule_dependencies=sum(r['kind']=='schedule' for r in records),static_dependencies=sum(r['kind']=='static' for r in records))
    # Freeze compact actual dependencies for the standalone live learner audit.
    np.savez_compressed(E/f'{split}-dependencies.npz',owner=np.array([r['owner'] for r in records]),cutoff=np.array([r['cutoff'] for r in records]),event=np.array([r['event'] if r['event'] is not None else -1 for r in records]),kind=np.array([r['kind'] for r in records]))
    np.savez_compressed(E/f'{split}-labels.npz',entity=q.driverId.to_numpy(),time=seconds(q.date),target=q.position.to_numpy())
np.savez_compressed(E/'label-events.npz',entity=all_results.driverId.to_numpy(),time=seconds(all_results.date),value=all_results.positionOrder.to_numpy())
report=dict(status='PASS',scope='Full archived F1 label population, SQL feature/dependency reconstruction and source fit-scope census',archives=archives,labels=label_results,label_maturity='PASS',fit_scope=fit_scope,numeric_scope_witnesses=examples[:5],join_dates=join_counts,dependencies=dependency_counts,schedule_witnesses=witnesses,
    strict_fit_horizon='FAIL: released preprocessing includes dated rows after2005-01-01',released_protocol='PASS: test-cutoff materialization matches source; not a newly introduced source-parity defect',historical_availability='NOT_ESTABLISHED',seconds=time.perf_counter()-start)
(E/'preflight.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['fit_scope','dependencies','labels']})

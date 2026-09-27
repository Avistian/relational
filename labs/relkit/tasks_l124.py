"""Entity task contracts. Explicit cohort choice preserves released semantics."""
# %% Imports (PROVIDED)
import json
import math
import pandas as pd

# %% Future labels (TODO 1)
def make_labels(results, cutoffs, cohort='released'):
    """Mean future position in (t,t+60d]; absent outcome produces no row.

    released: literal source eligibility date > t - one calendar year.
    past: additionally require eligibility observation date <= t.
    Neither mode invents a zero outcome for a driver with no future races.
    """
    if cohort not in ('released','past'):
        raise ValueError('Unknown cohort')
    cutoffs=pd.DatetimeIndex(cutoffs)
    if cutoffs.has_duplicates or cutoffs.isna().any():
        raise ValueError('Cutoffs must be unique and known')
    if results[['driverId','date','positionOrder']].isna().any().any():
        raise ValueError('Missing event identity/time/outcome')
    rows=[]
    for t in cutoffs:
        end=t+pd.Timedelta(days=60)
        active=results.date.gt(t-pd.DateOffset(years=1))
        if cohort=='past':active &= results.date.le(t)
        ids=set(results.loc[active,'driverId'])
        future=results.loc[results.date.gt(t)&results.date.le(end)&results.driverId.isin(ids)]
        for entity,value in future.groupby('driverId',sort=True).positionOrder.mean().items():
            rows.append(dict(driverId=int(entity),date=t,position=float(value),label_end=end))
    return pd.DataFrame(rows,columns=['driverId','date','position','label_end'])

# %% Task contract (TODO 2)
def validate_task(rows, entity_ids, fit_time=None):
    """Validate labeled queries; fit_time additionally requires mature labels."""
    required=['driverId','date','position','label_end']
    if not set(required).issubset(rows.columns) or rows[required].isna().any().any():
        raise ValueError('Task fields missing')
    if rows.duplicated(['driverId','date']).any():
        raise ValueError('Duplicate entity/time query')
    if not rows.driverId.isin(set(entity_ids)).all():
        raise ValueError('Dangling prediction entity')
    if not all(math.isfinite(float(v)) for v in rows.position):
        raise ValueError('Nonfinite target')
    if not rows.label_end.gt(rows.date).all():
        raise ValueError('Label horizon must end after query')
    if fit_time is not None and not rows.label_end.le(pd.Timestamp(fit_time)).all():
        raise ValueError('Labels not mature at fit time')
    return len(rows)

# %% Query-key alignment (TODO 3)
def aligned_mae(rows, predictions):
    """Score by (entity,time), never by an entity-only map or row coincidence."""
    keys=['driverId','date']
    for frame in (rows,predictions):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError('Invalid query identity')
    joined=rows.merge(predictions,on=keys,how='outer',validate='one_to_one',indicator=True)
    if len(joined)==0 or not joined['_merge'].eq('both').all():
        raise ValueError('Prediction coverage differs')
    if not all(math.isfinite(float(v)) for col in ['position','prediction'] for v in joined[col]):
        raise ValueError('Nonfinite score inputs')
    return math.fsum(abs(float(p)-float(y)) for p,y in zip(joined.prediction,joined.position))/len(joined)

# %% Course trace (PROVIDED)
def course_run():
    t=pd.Timestamp('2020-01-01')
    events=pd.DataFrame({'driverId':[0,0,0,1],'date':[t,t+pd.Timedelta(days=1),t+pd.Timedelta(days=60),t+pd.Timedelta(days=2)],'positionOrder':[99,2,6,3]})
    rows=make_labels(events,[t]);validate_task(rows,[0,1],t+pd.Timedelta(days=60))
    predictions=rows[['driverId','date']].iloc[::-1].copy();predictions['prediction']=[5.,5.]
    score=aligned_mae(rows,predictions)
    assert score==1.5
    return dict(status='PASS',released_rows=len(rows),past_rows=len(make_labels(events,[t],'past')),mae=score,learner_status='PENDING_WRITTEN_DEFENSE')

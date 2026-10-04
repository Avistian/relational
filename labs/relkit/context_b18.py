"""B18: exact historical sums, uniform sampling, and explicit information costs."""
import numpy as np


def eligible_history(rows,cutoff):
    """Keep credit events available at the cutoff; preserve stable event IDs."""
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate event IDs would double count evidence')
    if not np.isfinite(cutoff) or any(not np.isfinite(r['amount']) or not np.isfinite(r['day']) for r in rows):
        raise ValueError('Amounts and times must be finite')
    return [dict(r) for r in rows if r['day']<=cutoff and r['kind']=='credit']


def corrected_total(sample,population_size):
    """Unbiased total estimator only for a uniform sample without replacement."""
    sample=np.asarray(sample,dtype=float)
    if sample.ndim!=1 or not np.isfinite(sample).all():
        raise ValueError('Expected a finite one-dimensional sample')
    if not isinstance(population_size,(int,np.integer)) or population_size<0 or len(sample)>population_size:
        raise ValueError('Invalid eligible population size')
    if population_size==0:
        return 0.
    if len(sample)==0:
        raise ValueError('Cannot estimate a nonempty population from no observations')
    return float(population_size/len(sample)*sample.sum())


def stratified_metrics(degrees,predictions,targets):
    """Prediction-minus-target bias and RMSE within observed historical degree."""
    d=np.asarray(degrees);p=np.asarray(predictions,dtype=float);y=np.asarray(targets,dtype=float)
    if d.ndim!=1 or p.shape!=d.shape or y.shape!=d.shape or not len(d):
        raise ValueError('Aligned nonempty vectors required')
    if not (np.isfinite(d).all() and np.isfinite(p).all() and np.isfinite(y).all()) or (d<0).any() or (d!=d.astype(int)).any():
        raise ValueError('Finite values and nonnegative integral degrees required')
    out={}
    for degree in sorted(set(d.tolist())):
        errors=p[d==degree]-y[d==degree]
        out[str(int(degree))]=dict(n=len(errors),bias=float(errors.mean()),rmse=float(np.sqrt(np.mean(errors**2))))
    return out


def fixture(seed,degree,shape):
    """Fixed USD1200 historical mass; day31 future credit is an exclusion test."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,degree,shape=='concentrated',18]))
    values=np.full(degree,1200./degree)
    if shape=='concentrated':
        values[:]=240./(degree-1);values[0]=960.
    values=values[rng.permutation(degree)]
    rows=[dict(id=f'e{i}',day=i%28+1,month=1,kind='credit',amount=float(v)) for i,v in enumerate(values)]
    rows+=[dict(id='debit',day=15,month=1,kind='debit',amount=700.),dict(id='future',day=31,month=2,kind='credit',amount=1e9)]
    return rows,rng


def monthly_total(rows):
    """Aggregate every eligible event, retaining (month,event kind) groups."""
    groups={}
    for row in rows:
        key=(row['month'],row['kind'])
        groups[key]=groups.get(key,0.)+row['amount']
    return float(sum(groups.values()))


def run_experiment():
    """Execute the entire frozen paired grid; learner functions stay live."""
    fixtures=[];conditions=[]
    for seed in [0,1,2]:
        for degree in [8,64,512,4096]:
            for shape in ['uniform','concentrated']:
                rows,rng=fixture(seed,degree,shape)
                history=eligible_history(rows,30)
                values=np.array([r['amount'] for r in history]);truth=float(values.sum())
                # One prefix per repetition couples budget comparisons without looking at results.
                draws=[rng.permutation(degree)[:min(512,degree)].tolist() for _ in range(100)]
                changed=[dict(r,amount=-1e12) if r['id']=='future' else dict(r) for r in rows]
                assert eligible_history(changed,30)==history
                agg=monthly_total(history)
                fixtures.append(dict(seed=seed,degree=degree,shape=shape,rows=rows,draws=draws,truth=truth,future_invariant=True))
                for budget in [8,32,128,512]:
                    estimates={arm:[] for arm in ['exact','sampled','corrected','aggregate']}
                    for draw in draws:
                        sample=values[draw[:min(budget,degree)]]
                        estimates['exact'].append(truth)
                        estimates['sampled'].append(float(sample.sum()))
                        estimates['corrected'].append(corrected_total(sample,degree))
                        estimates['aggregate'].append(agg)
                    metrics={arm:stratified_metrics([degree]*100,pred,[truth]*100)[str(degree)] for arm,pred in estimates.items()}
                    conditions.append(dict(seed=seed,degree=degree,shape=shape,budget=budget,k=min(budget,degree),estimates=estimates,metrics=metrics))
    return dict(experiment='B18-CONTEXT-SUFFICIENCY',status='COMPLETE_MECHANISM_DIAGNOSTIC',fixtures=fixtures,conditions=conditions,cloud_usd=0,paper_status='INCOMPLETE_SOURCE_PROTOCOL_GATE',learner='PENDING_WRITTEN_DEFENSE')

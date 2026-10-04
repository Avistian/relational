"""B19b forecast-time contracts and explicit course diagnostic (not TabPFN)."""
import math,random
import numpy as np

def available_features(records, origin):
    """As-of lookup: each unique feature cell has an explicit arrival time."""
    result={};seen=set()
    for row in records:
        if not all(k in row for k in ('id','time','available_at','value')):
            raise ValueError('Feature cell requires identity, event time, availability and value')
        if row['id'] in seen:raise ValueError('Duplicate feature identity')
        seen.add(row['id'])
        if not math.isfinite(row['available_at']):raise ValueError('Invalid availability')
        if row['available_at']<=origin:result[row['id']]=row['value']
    return result

def rolling_origins(length, first, horizon, step):
    """Origin is the first forecast index; history is range(origin)."""
    if any(type(x) is not int or x<=0 for x in (length,first,horizon,step)) or first+horizon>length:
        raise ValueError('Require positive integers and a complete first horizon')
    return [(o,list(range(o,o+horizon))) for o in range(first,length-horizon+1,step)]

def forecast_scores(history, actual, quantiles, levels, season):
    """Median MASE and mean quantile loss; fail explicitly on zero denominators."""
    h=np.asarray(history,dtype=float);y=np.asarray(actual,dtype=float)
    p=np.asarray(quantiles,dtype=float);q=np.asarray(levels,dtype=float)
    if h.ndim!=1 or y.ndim!=1 or q.ndim!=1 or p.shape!=(len(y),len(q)) or not len(y):raise ValueError('Bad dimensions')
    if type(season) is not int or season<1 or len(h)<=season:raise ValueError('Insufficient seasonal history')
    if not all(np.isfinite(a).all() for a in (h,y,p,q)):raise ValueError('Nonfinite input')
    if not np.all((q>0)&(q<1)) or not np.all(np.diff(q)>0) or .5 not in q:raise ValueError('Ordered levels must include median')
    if not np.all(np.diff(p,axis=1)>=0):raise ValueError('Crossing quantiles')
    scale=float(np.mean(np.abs(h[season:]-h[:-season])));denom=float(np.abs(y).sum())
    if scale<=0 or denom<=0:raise ValueError('Undefined scale; declare benchmark-specific policy separately')
    error=y[:,None]-p;pinball=np.maximum(q*error,(q-1)*error)
    return dict(mase=float(np.abs(y-p[:,list(q).index(.5)]).mean()/scale),wql=float(2*pinball.sum(axis=0).mean()/denom),scale=scale)

def temporal_features(times):
    """Visible fixed course features. Not the full paper's feature extractor."""
    t=np.asarray(times,dtype=float)
    return np.column_stack([np.ones(len(t)),np.sin(2*np.pi*t/12),np.cos(2*np.pi*t/12),(t.astype(int)%24<3).astype(float)])

def fit_predict(history_times, history_y, query_times, history_weather=None, query_weather=None, query_promotion=None):
    """Fixed ridge + in-sample residual quantiles: teaching model, no calibration guarantee."""
    x=temporal_features(history_times);z=temporal_features(query_times)
    if query_promotion is not None:z[:,3]=query_promotion
    if history_weather is not None:
        x=np.column_stack([x,history_weather]);z=np.column_stack([z,query_weather])
    beta=np.linalg.solve(x.T@x+1e-6*np.eye(x.shape[1]),x.T@history_y)
    residual=history_y-x@beta
    return z@beta[:,None]+np.quantile(residual,np.arange(1,10)/10)[None,:]

def run_experiment():
    """Frozen 3 seeds x 3 origins x 3 arms x 12 steps; no selection."""
    levels=[i/10 for i in range(1,10)];series=[];predictions=[];scores=[];interventions=[]
    for seed in [0,1,2]:
        rng=random.Random(seed);weather=np.array([rng.gauss(0,1) for _ in range(156)])
        noise=np.array([rng.gauss(0,.5) for _ in range(156)]);t=np.arange(156)
        y=temporal_features(t)@np.array([20,3,0,2])+4*weather+noise
        series.append(dict(seed=seed,target=y.tolist(),weather=weather.tolist()))
        for origin,query in rolling_origins(156,96,12,24):
            # Issue immediately before origin. Calendar/promotion are announced 24 steps ahead.
            cells=[dict(id=f'{name}/{i}',time=i,available_at=i-24 if name=='promotion' else i,value=float((i%24<3) if name=='promotion' else weather[i])) for i in query for name in ['promotion','weather']]
            allowed=available_features(cells,origin-1)
            assert len(allowed)==12 and all(k.startswith('promotion/') for k in allowed)
            hist=y[:origin];qt=np.array(query)
            promotion=[allowed[f'promotion/{i}'] for i in query]
            legal=fit_predict(np.arange(origin),hist,qt,query_promotion=promotion)
            oracle=fit_predict(np.arange(origin),hist,qt,weather[:origin],weather[qt])
            differences=hist[12:]-hist[:-12]
            offsets=np.quantile(differences,levels);offsets-=offsets[4]
            naive=hist[origin-12:origin,None]+offsets[None,:]
            for arm,p in [('seasonal_naive',naive),('legal_regression',legal),('oracle_weather',oracle)]:
                score=forecast_scores(hist,y[qt],p,levels,12)
                scores.append(dict(seed=seed,origin=origin,arm=arm,**score))
                predictions.extend(dict(seed=seed,origin=origin,time=i,arm=arm,y=float(y[i]),quantiles=p[j].tolist()) for j,i in enumerate(query))
            mutated=[dict(c,value=c['value']+100) if c['id'].startswith('weather/') else c for c in cells]
            assert available_features(mutated,origin-1)==allowed
            # Legal prediction receives only immutable history and timestamps; no future target input.
            changed_allowed=available_features(mutated,origin-1)
            changed=fit_predict(np.arange(origin),hist,qt,query_promotion=[changed_allowed[f'promotion/{i}'] for i in query])
            interventions.append(dict(seed=seed,origin=origin,unavailable_feature_change_invariant=available_features(mutated,origin-1)==allowed,legal_predictions_invariant=bool(np.array_equal(changed,legal)),oracle_prediction_change=float(np.max(np.abs(fit_predict(np.arange(origin),hist,qt,weather[:origin],weather[qt]+100)-oracle)))))
    summary=[]
    for arm in ['seasonal_naive','legal_regression','oracle_weather']:
        by_seed=[]
        for seed in [0,1,2]:
            rows=[x for x in scores if x['arm']==arm and x['seed']==seed]
            by_seed.append(dict(seed=seed,**{m:sum(x[m] for x in rows)/len(rows) for m in ['mase','wql']}))
        summary.append(dict(arm=arm,by_seed=by_seed,**{m:sum(x[m] for x in by_seed)/3 for m in ['mase','wql']}))
    return dict(experiment='B19b-FORECAST-AVAILABILITY',status='COMPLETE_COURSE_DIAGNOSTIC',levels=levels,series=series,predictions=predictions,scores=scores,summary=summary,interventions=interventions,not_tabpfn=True)

def source_periods(history, k=5):
    """Mirror archived AutoSeasonalFeature.Config defaults; source/code differences documented."""
    from scipy import fft
    from scipy.signal import find_peaks
    x=np.asarray(history,dtype=float);x=x[~np.isnan(x)]
    if len(x)<3 or not np.isfinite(x).all():raise ValueError('Need at least three finite observations')
    t=np.arange(len(x));x=(x-np.polyval(np.polyfit(t,x,1,rcond=None),t))*np.hanning(len(x))
    magnitude=np.abs(fft.rfft(np.pad(x,(0,len(x)))));magnitude[0]=0
    frequencies=np.fft.rfftfreq(2*len(x))
    peaks,_=find_peaks(magnitude,height=.05*np.max(magnitude))
    if not len(peaks):peaks=np.arange(len(magnitude))
    indices=peaks[np.argsort(magnitude[peaks])[::-1]][:k]
    periods=np.zeros_like(frequencies);nonzero=frequencies>0;periods[nonzero]=1/frequencies[nonzero]
    top=np.round(periods[indices]);valid=top!=0;top=top[valid];indices=indices[valid]
    unique=np.unique(top,return_index=True)[1]
    return sorted([(float(top[i]),float(magnitude[indices[i]])) for i in unique],key=lambda x:x[1],reverse=True)

def source_seasonal_features(history, horizon):
    """History-only period fitting; future targets never enter this interface."""
    found=source_periods(history);periods=[x[0] for x in found];t=np.arange(len(history)+horizon)
    a=np.zeros((len(t),10))
    for i,p in enumerate(periods):a[:,2*i]=np.sin(2*np.pi*t/p);a[:,2*i+1]=np.cos(2*np.pi*t/p)
    return a[:len(history)],a[len(history):],periods

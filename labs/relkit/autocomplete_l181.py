"""Three live learner contracts used in the complete L181 audit."""
def visible_columns(columns, target, proxies, seed, policy):
    """Feature names only: time filtering remains a separate operation."""
    if policy not in ('global','seed_only') or len(set(columns))!=len(columns):
        raise ValueError('Unique columns and a recognized policy required')
    hidden=set([target,*proxies]) if policy=='global' or seed else set()
    return [c for c in columns if c not in hidden]

def baseline_predictions(fit, query, kind):
    """Released baseline recipe; caller explicitly chooses train or train+val."""
    import numpy as np
    y=fit['y'].to_numpy(dtype=float)
    if not len(y) or not np.isfinite(y).all():raise ValueError('Finite nonempty fit labels required')
    if kind=='global_zero':return np.zeros(len(query))
    if kind=='global_mean':return np.full(len(query),y.mean())
    if kind=='global_median':return np.full(len(query),np.median(y))
    if kind in ('entity_mean','entity_median'):
        grouped=fit.groupby('entity')['y'].agg('mean' if kind=='entity_mean' else 'median')
        return query['entity'].map(grouped).fillna(0).to_numpy(dtype=float)
    raise ValueError('Unknown baseline')

def keyed_scores(truth, predictions):
    """Join on the entire query identity, never positional order or entity alone."""
    import numpy as np
    keys=['entity','time']
    if any(d[keys].isna().any().any() or d.duplicated(keys).any() for d in [truth,predictions]):
        raise ValueError('Missing or duplicate query identity')
    if set(map(tuple,truth[keys].to_numpy()))!=set(map(tuple,predictions[keys].to_numpy())):
        raise ValueError('Prediction key set differs')
    joined=truth.merge(predictions,on=keys,validate='one_to_one')
    y=joined['y'].to_numpy(dtype=float);p=joined['pred'].to_numpy(dtype=float)
    if len(y)<2 or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Finite complete population required')
    sst=float(((y-y.mean())**2).sum())
    if sst==0:raise ValueError('R2 undefined for constant truth')
    return dict(n=len(y),mae=float(np.abs(y-p).mean()),r2=float(1-((y-p)**2).sum()/sst))

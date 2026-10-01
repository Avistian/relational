"""Visible context sampling, paired curve aggregation and evidence boundaries."""
def sample_context(n_rows,context,seed_key):
    import hashlib
    import numpy as np
    if type(n_rows) is not int or type(context) is not int or not 0<context<=n_rows or not isinstance(seed_key,str) or not seed_key:
        raise ValueError('Invalid context request')
    integer=int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4],'big')
    return np.random.default_rng(integer).choice(n_rows,context,replace=False)

def scaling_curve(records):
    import numpy as np
    contexts=[64,128,256,512,1024];seeds=list(range(10));values={}
    for r in records:
        k,s=r['context'],r['seed'];v=float(r['auc'])
        if type(k) is not int or type(s) is not int or k not in contexts or s not in seeds or (k,s) in values or not np.isfinite(v) or not 0<=v<=1:
            raise ValueError('Invalid or duplicate curve point')
        values[k,s]=v
    if set(values)!={(k,s) for k in contexts for s in seeds}:raise ValueError('Incomplete curve')
    levels=[];doublings=[]
    for i,k in enumerate(contexts):
        a=np.array([values[k,s] for s in seeds])
        levels.append(dict(context=k,per_seed=a.tolist(),mean=float(a.mean()),sample_sd=float(a.std(ddof=1))))
        if i:
            delta=a-np.array([values[contexts[i-1],s] for s in seeds])
            doublings.append(dict(start=contexts[i-1],end=k,per_seed=delta.tolist(),mean_gain=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),positive_seeds=int((delta>0).sum())))
    return dict(levels=levels,doublings=doublings,interpretation='SUPPORT_DRAW_VARIATION_ON_ONE_FIXED_TASK')

def scaling_claim(axis,controlled,levels,extrapolating):
    if axis not in ['context','parameters','pretraining_data','schema_diversity'] or type(controlled) is not bool or type(extrapolating) is not bool or type(levels) is not int or levels<1:
        raise ValueError('Invalid evidence declaration')
    if not controlled:return 'CONFOUNDED'
    if levels<2:return 'INSUFFICIENT_LEVELS'
    if extrapolating:return 'EXTRAPOLATION_NOT_ESTABLISHED'
    if axis=='context':return 'CONTEXT_RESPONSE_ONLY'
    return 'CONTROLLED_SWEEP_NOT_LAW'

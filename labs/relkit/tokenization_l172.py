"""Course typed-token contract. These are payloads, not learned RT embeddings."""
def fit_column(values, kind, admitted):
    """Fit only admitted nonnull values; empty numerical fit uses (0, 1)."""
    import numpy as np
    import pandas as pd
    if kind not in {'number','category','text','key','timestamp'}:
        raise ValueError('Unsupported semantic type: '+str(kind))
    s=pd.Series(values).reset_index(drop=True)
    flags=list(admitted)
    if len(flags)!=len(s) or any(not isinstance(v,(bool,np.bool_)) for v in flags):
        raise ValueError('Admission must be one boolean per row')
    train=s[pd.Series(flags,dtype=bool)].dropna()
    state=dict(kind=kind,fit_rows=int(sum(flags)),fit_nonnull=len(train))
    if kind=='number':
        try:a=np.asarray(train,dtype=float)
        except (ValueError,TypeError) as exc:raise ValueError('Expected numerical values') from exc
        if not np.isfinite(a).all():raise ValueError('Nonfinite numerical fit value')
        mean=float(a.mean()) if len(a) else 0.
        scale=float(a.std(ddof=0)) if len(a) else 1.
        state.update(mean=mean,scale=scale if scale>0 else 1.)
    if kind=='category':state['vocabulary']=sorted({str(v) for v in train})
    return state


def encode_column(values, fitted, masked=None):
    """State precedes payload: masked values are erased even when originally null."""
    import numpy as np
    import pandas as pd
    s=pd.Series(values).reset_index(drop=True);kind=fitted['kind']
    if kind not in {'number','category','text','key','timestamp'}:raise ValueError('Unsupported semantic type')
    mask=[False]*len(s) if masked is None else list(masked)
    if len(mask)!=len(s) or any(not isinstance(v,(bool,np.bool_)) for v in mask):
        raise ValueError('Mask must be one boolean per row')
    flags=np.asarray(mask,dtype=bool);missing=s.isna().to_numpy();active=~(flags|missing)
    states=np.full(len(s),'VALUE',dtype=object);states[missing]='MISSING';states[flags]='MASKED'
    neutral='' if kind in {'key','text'} else 0.
    payload=np.full(len(s),neutral,dtype=object)
    observed=s[active]
    if kind=='number':
        try:a=np.asarray(observed,dtype=float)
        except (ValueError,TypeError) as exc:raise ValueError('Expected numerical values') from exc
        if not np.isfinite(a).all():raise ValueError('Nonfinite numerical transform value')
        if not np.isfinite(fitted['mean']) or not np.isfinite(fitted['scale']) or fitted['scale']<=0:raise ValueError('Invalid fitted normalization')
        payload[active]=(a-fitted['mean'])/fitted['scale']
    elif kind=='timestamp':
        dates=pd.to_datetime(observed,utc=True,errors='raise')
        payload[active]=((dates-pd.Timestamp('1970-01-01',tz='UTC'))/pd.Timedelta(days=1)).to_numpy()
    elif kind=='category':
        vocab={v:i+1 for i,v in enumerate(fitted['vocabulary'])}
        codes=[vocab.get(str(v),0) for v in observed]
        payload[active]=codes
        positions=np.flatnonzero(active)
        states[positions[np.asarray(codes)==0]]='UNKNOWN'
    else:
        # Never cast key identifiers through floating point: 2**53+1 must survive.
        payload[active]=[str(v) for v in observed]
    return dict(state=states.tolist(),payload=payload.tolist())


def tokenize_table(frame, schema, admitted, masks=None):
    """Canonical column names survive reordering; key roles remain explicit."""
    if not frame.columns.is_unique or set(frame.columns)!=set(schema):
        raise ValueError('Schema must declare every column exactly once')
    masks={} if masks is None else masks
    if set(masks)-set(schema):raise ValueError('Mask names an absent column')
    columns={}
    for name in sorted(schema):
        spec=schema[name]
        if spec.get('role') not in {'feature','primary_key','foreign_key','event_time'}:
            raise ValueError('Unsupported column role')
        if spec['role'] in {'primary_key','foreign_key'} and spec['kind']!='key':
            raise ValueError('Keys must preserve identity, not magnitude')
        fitted=fit_column(frame[name],spec['kind'],admitted)
        columns[name]=dict(schema=dict(spec),fitted=fitted,**encode_column(frame[name],fitted,masks.get(name)))
    return dict(rows=len(frame),columns=columns)

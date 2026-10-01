"""Visible row-encoding and split-safe selection operators (no global fitting)."""
def serialize_row(row, variant='baseline'):
    """Allow-list the four inputs; target and row identity never enter text."""
    import json, math
    columns=['price_usd','weight_kg','colour','condition']
    if not isinstance(row,dict) or not all(k in row for k in columns):
        raise ValueError('Four canonical input fields required')
    for k in columns[:2]:
        if isinstance(row[k],bool) or not isinstance(row[k],(int,float)) or not math.isfinite(row[k]):
            raise ValueError('Numeric fields must be finite numbers')
    if any(not isinstance(row[k],str) or not row[k] for k in columns[2:]):
        raise ValueError('Categories must be nonempty strings')
    if variant not in ['baseline','reordered','renamed']:raise ValueError('Unknown intervention')
    ordered=list(reversed(columns)) if variant=='reordered' else columns
    payload={'table':'products'}
    for k in ordered:
        name='c'+str(columns.index(k)) if variant=='renamed' else k
        payload[name]=row[k]
    return json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def masked_mean(hidden, mask):
    """Average final token vectors over real tokens, including BOS/EOS, not PAD."""
    import numpy as np
    h=np.asarray(hidden);m=np.asarray(mask)
    if h.ndim!=3 or m.shape!=h.shape[:2] or not np.isfinite(h).all():
        raise ValueError('Expected finite [batch,tokens,width] and aligned mask')
    if not np.isin(m,[0,1]).all() or (m.sum(axis=1)==0).any():
        raise ValueError('Binary mask with at least one included token per row required')
    return (h*m[:,:,None]).sum(axis=1)/m.sum(axis=1)[:,None]


def choose_alpha(alphas, validation_mae):
    """Only validation losses enter this interface; ties retain first grid entry."""
    import numpy as np
    a=np.asarray(alphas,dtype=float);v=np.asarray(validation_mae,dtype=float)
    if a.ndim!=1 or not len(a) or a.shape!=v.shape or not np.isfinite(a).all() or not np.isfinite(v).all() or (a<=0).any() or (v<0).any():
        raise ValueError('Positive alphas and aligned finite nonnegative validation errors required')
    return float(a[int(np.argmin(v))])


def typed_features(train_rows, rows):
    """Train-only numeric moments plus train-only one-hot vocabularies."""
    import numpy as np
    if not train_rows or not rows:raise ValueError('Nonempty training and query rows required')
    for row in train_rows+rows:serialize_row(row)
    numeric=['price_usd','weight_kg'];categorical=['colour','condition']
    train=np.asarray([[r[k] for k in numeric] for r in train_rows],dtype=float)
    mean=train.mean(axis=0);scale=train.std(axis=0);scale[scale==0]=1.
    vocab={k:sorted({r[k] for r in train_rows}) for k in categorical}
    numbers=(np.asarray([[r[k] for k in numeric] for r in rows])-mean)/scale
    onehot=np.asarray([[float(r[k]==v) for k in categorical for v in vocab[k]] for r in rows])
    return np.concatenate([numbers,onehot],axis=1),dict(mean=mean.tolist(),scale=scale.tolist(),categories=vocab)

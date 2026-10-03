"""B07 semantic interventions, validation-only probe and paired evidence."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def intervene(frame, arm):
    """Remove natural header meanings; numeric-only also removes text columns."""
    if arm not in ('meaningful','anonymous','numeric_only'):
        raise ValueError('Unknown arm')
    if not frame.columns.is_unique:
        raise ValueError('Duplicate feature names')
    out=frame.copy(deep=True)
    if arm=='meaningful':
        return out
    labels=[str(i) for i in range(6,19)]+['20','0']
    if len(frame.columns)>len(labels):
        raise ValueError('Anonymous label cache covers at most 15 features')
    mapping=dict(zip(frame.columns,labels))
    if arm=='numeric_only':
        out=out.select_dtypes(include='number')
    return out.rename(columns=mapping)


def fit_probe(features, target, train, validation, test, alphas=(1.,10.,100.)):
    """Fit on train, select on validation, predict test; retain full head state."""
    x=np.asarray(features,dtype=float);y=np.asarray(target,dtype=float)
    groups=[np.asarray(ids,dtype=int) for ids in (train,validation,test)]
    flat=np.concatenate(groups)
    if any(len(ids)==0 for ids in groups) or len(np.unique(flat))!=len(flat):
        raise ValueError('Nonempty disjoint split identities required')
    if flat.min()<0 or flat.max()>=len(x) or len(y)!=len(x):
        raise ValueError('Invalid row identities')
    scaler=StandardScaler().fit(x[train])
    tr,va,te=[scaler.transform(x[ids]) for ids in groups]
    candidates=[]
    for alpha in alphas:
        model=Ridge(alpha=alpha).fit(tr,y[train])
        loss=float(np.mean((model.predict(va)-y[validation])**2))
        candidates.append((loss,float(alpha),model))
    selected=min(range(len(candidates)),key=lambda i:candidates[i][0])
    loss,alpha,model=candidates[selected]
    return dict(alpha=alpha,prediction=model.predict(te).tolist(),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficient=model.coef_.tolist(),intercept=float(model.intercept_),validation_mse=[v[0] for v in candidates])


def paired_delta(left, right):
    """Return left-minus-right, pairing complete table/seed keys, never positions."""
    if not left or set(left)!=set(right):
        raise ValueError('Identical nonempty identity sets required')
    return [float(left[k]-right[k]) for k in sorted(left)]


def encode_table(model,frame,vectors,graph_fn,embed_fn):
    """Preserve all rows; no observed leaf maps to a declared zero representation."""
    records=frame.to_dict('records')
    observed=[i for i,row in enumerate(records) if any(not pd.isna(v) for v in row.values())]
    out=np.zeros((len(frame),300),dtype=np.float32)
    if observed:
        graphs=[graph_fn(records[i],vectors) for i in observed]
        out[observed]=embed_fn(model,graphs)
    return out

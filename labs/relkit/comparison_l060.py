"""L060 v2: live audit/selection/aggregation, lower-is-better metrics throughout."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, friedmanchisquare, studentized_range, t


def validate_partitions(ids):
    if set(ids) != {'train','val','test'}:
        raise ValueError('Require exactly train, val, test')
    arrays=[np.asarray(ids[s]) for s in ['train','val','test']]
    if any(a.ndim!=1 or not len(a) or not np.issubdtype(a.dtype,np.integer) for a in arrays):
        raise ValueError('Require nonempty one-dimensional integer row IDs')
    joined=np.concatenate(arrays)
    if len(np.unique(joined))!=len(joined):
        raise ValueError('Duplicate row within or across partitions')
    return True


def choose_validation(errors):
    errors=np.asarray(errors,float)
    if errors.ndim!=1 or not errors.size or not np.isfinite(errors).all():
        raise ValueError('Require finite nonempty validation errors')
    return int(np.argmin(errors))


def score_predictions(y,p,regression):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if y.ndim!=1 or p.shape!=y.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Require aligned finite nonempty vectors')
    if not isinstance(regression,(bool,np.bool_)):raise ValueError('Declare regression explicitly')
    if regression:return float(np.sqrt(np.mean((y-p)**2)))
    if not np.isin(y,[0,1]).all() or np.any((p<0)|(p>1)):
        raise ValueError('Binary labels and P(y=1) in [0,1] required')
    p=np.clip(p,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def aggregate_panel(records,datasets,arms,seeds):
    """Explicit expected design detects even an entirely absent dataset or arm."""
    if not records or any(not len(v) or len(set(v))!=len(v) for v in [datasets,arms,seeds]):
        raise ValueError('Declare unique nonempty datasets, arms and seeds')
    f=pd.DataFrame(records)
    key=['dataset','arm','seed']
    if not set(key+['error','seconds']).issubset(f.columns) or f.duplicated(key).any():
        raise ValueError('Missing fields or duplicated result')
    expected=pd.MultiIndex.from_product([datasets,arms,seeds],names=key)
    indexed=f.set_index(key)
    if set(indexed.index)!=set(expected):raise ValueError('Incomplete or unexpected crossed panel')
    if not np.isfinite(f[['error','seconds']].to_numpy(float)).all() or (f.seconds<0).any():
        raise ValueError('Require finite error and nonnegative recorded cost')
    table=indexed.error.groupby(['dataset','arm']).mean().unstack('arm').reindex(index=datasets,columns=arms)
    ranks=np.stack([rankdata(row,method='average') for row in table.to_numpy()])
    n,k=ranks.shape
    p=float(friedmanchisquare(*table.to_numpy().T).pvalue) if n>=3 and k>=3 and np.any(np.ptp(ranks,axis=1)) else None
    cd=float(studentized_range.ppf(.95,k,np.inf)*np.sqrt(k*(k+1)/(12*n))) if k>1 else None
    details=[]
    for d in datasets:
        for a in arms:
            g=indexed.loc[(d,a)].reindex(seeds);v=g.error.to_numpy();sd=float(v.std(ddof=1)) if len(v)>1 else None
            half=float(t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))) if sd is not None else None
            details.append(dict(dataset=d,arm=a,mean=float(v.mean()),sd=sd,seed_values=v.tolist(),conditional_t95=None if half is None else [float(v.mean()-half),float(v.mean()+half)],seconds=float(g.seconds.sum())))
    return dict(datasets=n,seeds=list(seeds),arms=list(arms),details=details,mean_ranks=dict(zip(arms,ranks.mean(0).tolist())),dataset_ranks=dict(zip(datasets,ranks.tolist())),friedman_p=p,nemenyi_cd=cd,uncertainty='Conditional model-seed variation; dataset is the ranking block')


def paired_effect(records,dataset,arm,baseline):
    f=pd.DataFrame(records);f=f[(f.dataset==dataset)&f.arm.isin([arm,baseline])]
    if arm==baseline or f.duplicated(['arm','seed']).any():raise ValueError('Require distinct arms and unique seeds')
    a=f[f.arm==arm].set_index('seed').error;b=f[f.arm==baseline].set_index('seed').error
    if len(a)<2 or set(a.index)!=set(b.index):raise ValueError('Require at least two matching seed pairs')
    delta=(a.sort_index()-b.sort_index()).to_numpy(float)
    if not np.isfinite(delta).all():raise ValueError('Nonfinite paired effect')
    mean=float(delta.mean());sd=float(delta.std(ddof=1));half=float(t.ppf(.975,len(delta)-1)*sd/np.sqrt(len(delta)))
    return dict(n=len(delta),mean=mean,sd=sd,t95=[mean-half,mean+half],differences=delta.tolist(),direction='Negative favors named arm; conditional on fixed split')


def audit_records(result):
    """Reconstruct predictions and test identity before presenting any result."""
    for audit in result['datasets'].values():validate_partitions(audit['ids'])
    target_by_dataset={}
    for r in result['records']:
        a=result['datasets'][r['dataset']]
        if r['test_ids']!=a['ids']['test']:raise ValueError('Prediction row order differs from declared test IDs')
        if r['dataset'] in target_by_dataset and target_by_dataset[r['dataset']]!=r['targets']:
            raise ValueError('Different targets across supposedly paired arms or seeds')
        target_by_dataset[r['dataset']]=r['targets']
        if len(r['targets'])!=len(r['test_ids']):raise ValueError('Target/ID length mismatch')
        if r['selected']!=choose_validation(r['validation_errors']):raise ValueError('Choice differs from validation minimum')
        if not np.isclose(r['error'],score_predictions(r['targets'],r['predictions'],a['regression']),atol=1e-12,rtol=1e-10):
            raise ValueError('Saved metric differs from predictions')
    summaries={}
    for regime,ds in result['design']['panels'].items():
        rows=[r for r in result['records'] if r['dataset'] in ds]
        summaries[regime]=aggregate_panel(rows,ds,result['design']['arms'],result['config']['seeds'])
    expected=set(sum(result['design']['panels'].values(),[]))
    if set(result['datasets'])!=expected or any(r['dataset'] not in expected for r in result['records']):
        raise ValueError('Undeclared or missing dataset')
    return summaries

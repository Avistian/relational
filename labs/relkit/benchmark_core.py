"""Inspectible selection, aggregation and negative-control algorithms, L058–070."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, friedmanchisquare, studentized_range, t


def choose_validation(errors):
    errors=np.asarray(errors,float)
    if errors.ndim!=1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError('Expected finite validation errors')
    return int(np.argmin(errors))


def validate_partitions(ids):
    if set(ids)!={'train','val','test'}:
        raise ValueError('Require train, val and test IDs')
    arrays=[np.asarray(v) for v in ids.values()]
    joined=np.concatenate(arrays)
    if not all(len(v)>0 for v in arrays) or len(np.unique(joined))!=len(joined):
        raise ValueError('Partitions must be nonempty, duplicate-free and disjoint')
    return True


def paired_summary(records):
    """Average seeds per dataset BEFORE ranks; reject an incomplete crossed design."""
    frame=pd.DataFrame(records)
    if frame.duplicated(['dataset','arm','seed']).any():raise ValueError('Duplicate result')
    datasets,arms,seeds=[sorted(frame[c].unique()) for c in ['dataset','arm','seed']]
    expected=pd.MultiIndex.from_product([datasets,arms,seeds],names=['dataset','arm','seed'])
    wide=frame.set_index(['dataset','arm','seed']).reindex(expected)
    if wide.error.isna().any() or not np.isfinite(wide.error).all():raise ValueError('Incomplete paired design')
    table=wide.error.groupby(['dataset','arm']).mean().unstack('arm')
    ranks=np.array([rankdata(row,method='average') for row in table.to_numpy()])
    n,k=ranks.shape
    if n>=3 and k>=3 and np.any(np.ptp(ranks,axis=1)):
        p=float(friedmanchisquare(*table.to_numpy().T).pvalue)
    else:p=None
    cd=float(studentized_range.ppf(.95,k,np.inf)*np.sqrt(k*(k+1)/(12*n))) if k>=2 else None
    details=[]
    for (dataset,arm),g in frame.groupby(['dataset','arm']):
        values=g.sort_values('seed').error.to_numpy();sd=float(values.std(ddof=1)) if len(values)>1 else 0.
        half=float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))) if len(values)>1 else None
        details.append(dict(dataset=dataset,arm=arm,mean=float(values.mean()),sd=sd,
            seed_values=values.tolist(),conditional_t95=None if half is None else [float(values.mean()-half),float(values.mean()+half)],
            seconds=float(g.seconds.sum())))
    return dict(datasets=n,seeds=[int(s) for s in seeds],arms=arms,details=details,
        mean_ranks=dict(zip(arms,ranks.mean(0).tolist())),friedman_p=p,nemenyi_cd=cd,
        uncertainty='Conditional model-seed variation on fixed rows; datasets are the ranking blocks')


def null_search(candidates,n_validation,n_test,seed=59):
    """Pure-noise candidates; separate RNG streams prevent test-dependent selection."""
    val_rng=np.random.default_rng(seed);test_rng=np.random.default_rng(seed+10000)
    vy=val_rng.integers(0,2,n_validation);ty=test_rng.integers(0,2,n_test)
    vp=val_rng.integers(0,2,(candidates,n_validation))
    errors=(vp!=vy).mean(1);selected=choose_validation(errors)
    # No test predictions generated until the selected index is frozen.
    tp=test_rng.integers(0,2,(candidates,n_test))[selected]
    return dict(selected=selected,validation_error=float(errors[selected]),
        test_error=float((tp!=ty).mean()),candidates=candidates)


def dataset_bootstrap(matrix,draws=1000,seed=58):
    """Paired bootstrap of dataset-level model scores/ranks; rows are the unit."""
    x=np.asarray(matrix,float)
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)
    return np.quantile(means,[.025,.975],axis=0)


def select_ensemble_prefix(predictions,labels,max_steps):
    """Uses the existing L057 greedy selector; all adaptive steps count as selection."""
    from relkit.cross_ensemble import greedy_select
    return greedy_select(predictions,labels,max_steps)


def attention_cost(n_context,n_query,features,inducing,width=128,bytes_per_value=4):
    """Illustrative score elements per head/layer, NOT peak memory or runtime.

    Axial: N*F² + F*N*C. TabICL: F*m*(C+N) + N*F² + N*C.
    Omits CLS/target/grouping constants, QKV/FFN, layers, caching and fused kernels.
    """
    n=n_context+n_query
    axial=n*features**2+features*n*n_context
    column=features*inducing*(n_context+n)
    row=n*features**2
    icl=n*n_context
    return dict(axial_elements=axial,tabicl_column_elements=column,
                tabicl_row_elements=row,tabicl_icl_elements=icl,
                axial_gib=axial*bytes_per_value/2**30,
                tabicl_gib=(column+row+icl)*bytes_per_value/2**30)

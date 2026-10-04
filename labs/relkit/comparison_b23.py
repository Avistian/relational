"""Visible B23 comparison contracts; model computation is inherited from L166."""
import numpy as np

def admit_grid(records):
    """Require four arms × ten draws; no survivor-only average is admissible."""
    arms={'RDBPFN','RDBPFN_single','TabICLv1.1','Logistic'}
    expected={(a,s) for a in arms for s in range(10)};seen=set()
    for row in records:
        if not isinstance(row,dict) or not {'arm','seed','rows','support','auc'}<=row.keys():raise ValueError('Missing fields')
        if not isinstance(row['arm'],str) or any(type(row[k]) is not int for k in ['seed','rows','support']):raise ValueError('Invalid types')
        key=(row['arm'],row['seed'])
        if key not in expected or key in seen or row['rows']!=702 or row['support']!=512:raise ValueError('Wrong grid')
        if not np.isfinite(row['auc']) or not 0<=row['auc']<=1:raise ValueError('Invalid AUROC')
        seen.add(key)
    if seen!=expected:raise ValueError('Incomplete comparison')
    return dict(runs=40,predictions=28080,paper_runs=30,course_runs=10)

def paired_comparison(records, reference, challenger):
    """Challenger minus reference on the SAME support draw and fixed test set."""
    admit_grid(records)
    arms={r['arm'] for r in records}
    if reference not in arms or challenger not in arms or reference==challenger:raise ValueError('Invalid comparison')
    lookup={(r['arm'],r['seed']):r['auc'] for r in records}
    delta=np.array([lookup[challenger,s]-lookup[reference,s] for s in range(10)])
    return dict(reference=reference,challenger=challenger,seeds=list(range(10)),per_seed=delta.tolist(),mean=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),positive=int((delta>0).sum()))

def claim_status(complete, close, historical, available):
    """Numerical closeness cannot fill missing provenance or learner evidence."""
    if any(type(x) is not bool for x in [complete,close,historical,available]):raise ValueError('Boolean evidence required')
    return dict(numerical=('CLOSE' if close else 'OUTSIDE_TOLERANCE') if complete else 'INCOMPLETE',historical='ESTABLISHED' if historical else 'NOT_ESTABLISHED',deployment_validity='ESTABLISHED' if available else 'NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

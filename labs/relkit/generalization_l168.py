"""Contracts for cross-database evidence; no training or cloud dispatch."""
def transfer_regime(pretraining_databases, target_database, exposure_complete, support_labels, gradient_steps):
    """Classify database exposure separately from target-task adaptation.

    A complete exposure list is an input assumption, not inferred from a model name.
    Synthetic schema reuse does not by itself imply reuse of target database rows.
    """
    if not isinstance(target_database,str) or not target_database or type(exposure_complete) is not bool:
        raise ValueError('Explicit target and exposure completeness required')
    if isinstance(pretraining_databases,str):raise ValueError('Expected database identifiers, not a string')
    pretraining_databases=list(pretraining_databases)
    if any(not isinstance(x,str) or not x for x in pretraining_databases):
        raise ValueError('Expected a collection of database identifiers')
    if any(type(x) is not int or x<0 for x in [support_labels,gradient_steps]):
        raise ValueError('Label and update counts must be nonnegative integers')
    if gradient_steps and not support_labels:raise ValueError('This contract covers supervised adaptation only')
    seen=target_database in pretraining_databases
    holdout='SEEN' if seen else ('HELD_OUT' if exposure_complete else 'NOT_ESTABLISHED')
    adaptation='SUPERVISED_ADAPTATION' if gradient_steps else ('FEW_SHOT_ICL' if support_labels else 'ZERO_LABEL_ZERO_GRADIENT')
    return dict(database_holdout=holdout,adaptation=adaptation)


def paired_gains(records, databases, seeds):
    """Exactly one selected task per database; pair by database, seed and model.

    Population/support identities must already have passed the raw-file audit.
    This function additionally enforces complete paired AUROC and row-count records.
    """
    import numpy as np
    if not databases or not seeds or len(set(databases))!=len(databases) or len(set(seeds))!=len(seeds):
        raise ValueError('Unique nonempty databases and seeds required')
    if any(type(s) is not int or s<0 for s in seeds):raise ValueError('Invalid seed')
    expected={(d,s,a) for d in databases for s in seeds for a in ['RDBPFN','TabICLv1.1']}
    lookup={};counts={}
    for r in records:
        key=(r['database'],r['seed'],r['arm'])
        if key not in expected or key in lookup:raise ValueError('Unexpected or duplicate run')
        if r['metric']!='AUROC' or not np.isfinite(r['auc']) or not 0<=r['auc']<=1:
            raise ValueError('Expected finite AUROC in [0,1]')
        n=r['test_rows'];d=r['database']
        if type(n) is not int or n<=0 or (d in counts and counts[d]!=n):raise ValueError('Unmatched test populations')
        counts[d]=n;lookup[key]=float(r['auc'])
    if set(lookup)!=expected:raise ValueError('Missing paired run')
    output={}
    for d in databases:
        delta=np.array([lookup[d,s,'RDBPFN']-lookup[d,s,'TabICLv1.1'] for s in sorted(seeds)])
        output[d]=dict(per_seed=delta.tolist(),mean=float(delta.mean()),
                       sample_sd=float(delta.std(ddof=1)) if len(delta)>1 else None,
                       positive_seeds=int((delta>0).sum()),test_rows=counts[d],seeds=sorted(seeds),metric='AUROC')
    return output


def database_macro(gains):
    """Equal weight per database, never per row or support draw.

    Describes these selected databases; supplies no population-level confidence claim.
    """
    import numpy as np
    if not gains:raise ValueError('No database evidence')
    values=[]
    for d,r in gains.items():
        v=np.asarray(r['per_seed'],dtype=float)
        if r['metric']!='AUROC' or v.ndim!=1 or not len(v) or not np.isfinite(v).all() or np.any(np.abs(v)>1):
            raise ValueError('Invalid per-database gains')
        if not np.isclose(v.mean(),r['mean'],rtol=0,atol=1e-12):raise ValueError('Inconsistent database mean')
        values.append(float(v.mean()))
    return dict(databases=len(values),macro_gain=float(np.mean(values)),positive_databases=int(np.sum(np.array(values)>0)),
                inference='DESCRIPTIVE_SELECTED_DATABASES_ONLY')

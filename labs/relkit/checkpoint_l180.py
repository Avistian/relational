"""Live checkpoint contracts; these functions never dispatch training."""

def temporal_counts(timestamps,padding,targets,label_cells,masks,cutoff,horizon):
    """Audit source event times; unknown arrival times remain unresolved."""
    import numpy as np
    ts=np.asarray(timestamps,dtype=np.int64)
    pad,target,label,mask=[np.asarray(x,dtype=bool) for x in (padding,targets,label_cells,masks)]
    if ts.ndim!=1 or any(x.shape!=ts.shape for x in (pad,target,label,mask)) or horizon<=0:
        raise ValueError('One-dimensional equal shapes and positive horizon required')
    known=ts!=np.iinfo(np.int32).min
    valid=~pad;visible=valid&label&~mask
    return dict(future_cells=int((valid&known&(ts>cutoff)).sum()),
                unmasked_query_targets=int((valid&target&~mask).sum()),
                unavailable_labels=int((visible&known&(ts+horizon>cutoff)).sum()),
                unknown_time_cells=int((valid&~known).sum()))

def full_run_cost(gpus,hours,rate_per_second,runs=1,overhead=0):
    """Decimal USD scenario: reported wall time times every allocated GPU."""
    from decimal import Decimal,InvalidOperation
    try:
        values=[Decimal(str(x)) for x in (gpus,hours,rate_per_second,runs,overhead)]
    except (InvalidOperation,ValueError):raise ValueError('Invalid cost input') from None
    g,h,r,n,o=values
    if not all(x.is_finite() for x in values) or min(g,h,r,n)<=0 or o<0 or g!=int(g) or n!=int(n):
        raise ValueError('Finite positive counts, time and rate required')
    return format(g*h*Decimal(3600)*r*n+o,'.6f')

def checkpoint_decision(evidence):
    """Retain all failed prerequisites; hypothetical admission cannot award mastery."""
    from decimal import Decimal,InvalidOperation
    blockers=[]
    for field in ['temporal','training_health','checkpoint_bytes','selection','full_population','source_protocol']:
        if evidence.get(field)!='PASS':blockers.append(field.upper())
    try:
        cost=Decimal(str(evidence.get('gpu_only_usd')));cap=Decimal(str(evidence.get('cap_usd')))
        if not cost.is_finite() or not cap.is_finite() or cost<0 or cap<=0:raise ValueError()
        if cost>cap:blockers.append('BUDGET')
    except (InvalidOperation,ValueError):blockers.append('BUDGET_UNKNOWN')
    try:
        total=Decimal(str(evidence.get('all_in_upper_usd')))
        if not total.is_finite() or total<0:raise ValueError()
        if 'BUDGET_UNKNOWN' not in blockers:
            if total<cost:blockers.append('ALL_IN_COST_INVALID')
            elif total>cap and 'BUDGET' not in blockers:blockers.append('BUDGET')
    except (InvalidOperation,ValueError):blockers.append('ALL_IN_COST_UNKNOWN')
    admission='BLOCKED' if blockers else 'READY_FOR_SEPARATELY_AUTHORIZED_RUN'
    practical='INCOMPLETE'
    if not blockers and evidence.get('fresh_fit')=='VERIFIED':
        practical='PASS' if evidence.get('written_defense')=='PASS' else 'PENDING_WRITTEN_DEFENSE'
    return dict(admission=admission,blockers=blockers,practical_exit=practical)

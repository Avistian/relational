"""Pure claim contracts for a bounded thesis verdict."""
import math

def evidence_coverage(rows):
    """Count tasks/databases once; identical prediction bytes are reused evidence."""
    seen=set();tasks=set();databases=set();missing=set();packets={}
    for row in rows:
        if not row.get('id') or row['id'] in seen:raise ValueError('Evidence IDs must be unique')
        seen.add(row['id'])
        if row.get('split')!='test' or row.get('status') not in ('COMPLETE','INCOMPLETE','NOT_RUN'):
            raise ValueError('Coverage requires explicit test status')
        if not row.get('task') or not row.get('database'):raise ValueError('Declare task and database')
        if row['status']=='COMPLETE':
            if not row.get('prediction_hash'):raise ValueError('Completed evidence needs prediction identity')
            tasks.add(row['task']);databases.add(row['database'])
            packets.setdefault(row['prediction_hash'],[]).append(row['id'])
        else:missing.add(row['task'])
    return dict(completed_tasks=len(tasks),completed_databases=len(databases),
                unique_prediction_packets=len(packets),reused_packets=[sorted(v) for v in packets.values() if len(v)>1],
                missing_tasks=sorted(missing-tasks),independence='NOT_ESTABLISHED_BY_FILE_COUNT')

def claim_verdict(claim,evidence):
    """A deterministic scope guard, not a statistical proof or automatic essay grade."""
    if claim=='local_quality':
        b=evidence['benefit'];lo,hi=evidence['interval']
        if not all(math.isfinite(x) for x in [b,lo,hi]) or lo>hi:raise ValueError('Invalid effect/interval')
        if evidence['matched_tasks']<1:return 'NOT_ESTABLISHED'
        direction='RDL' if b>0 else 'FE' if b<0 else 'NEITHER'
        boundary='SUPERIORITY_NOT_ESTABLISHED' if lo<=0<=hi else 'CONDITIONAL_INTERVAL_EXCLUDES_ZERO'
        return 'POINT_ESTIMATE_FAVORS_'+direction+'; '+boundary
    if claim=='portfolio_superiority':
        # This lesson has no cross-database sampling/inference protocol.
        return 'NOT_ESTABLISHED'
    if claim=='human_effort':return evidence['human_effort']
    if claim=='leak_free':return evidence['historical_availability']
    raise ValueError('Unknown claim')

def validate_falsifier(spec):
    """Check an operational future test; execution and scientific adequacy need review."""
    for k in ('task','metric','direction','baseline','information_policy','disconfirm_if'):
        if not isinstance(spec.get(k),str) or not spec[k].strip():raise ValueError('Missing '+k)
    if spec['metric'] not in ('MAE','AUROC','MAP@10') or spec['direction']!=('lower' if spec['metric']=='MAE' else 'higher'):
        raise ValueError('Metric direction mismatch')
    x=spec.get('minimum_benefit')
    if not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:raise ValueError('Invalid threshold')
    if spec.get('selection_split')!='val' or spec.get('decision_split')!='test':raise ValueError('Freeze on validation; evaluate on test')
    return dict(spec,status='SPECIFIED_NOT_EXECUTED',review='PENDING_SCIENTIFIC_REVIEW')

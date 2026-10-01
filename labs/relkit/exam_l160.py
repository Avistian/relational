"""Visible Year 4 exam mechanisms."""
import math
import numpy as np

def aligned_losses(truth_keys, targets, fe_keys, fe_predictions, rdl_keys, rdl_predictions):
    """Pair absolute errors by complete (entity, cutoff) identity, in truth order."""
    packets=[]
    for keys,values in [(truth_keys,targets),(fe_keys,fe_predictions),(rdl_keys,rdl_predictions)]:
        keys=[tuple(k) for k in keys]
        if not keys or any(len(k)!=2 for k in keys) or len(keys)!=len(values) or len(set(keys))!=len(keys):
            raise ValueError('Nonempty unique query keys and one value per key required')
        if not all(math.isfinite(float(v)) for v in values): raise ValueError('Nonfinite value')
        packets.append(dict(zip(keys,map(float,values))))
    truth,fe,rdl=packets
    if set(truth)!=set(fe) or set(truth)!=set(rdl): raise ValueError('Different query populations')
    a=[abs(truth[k]-fe[k]) for k in truth];b=[abs(truth[k]-rdl[k]) for k in truth]
    benefit=[f-g for f,g in zip(a,b)]
    return dict(fe_loss=a,rdl_loss=b,benefit=benefit,mean_benefit=sum(benefit)/len(benefit))

def observed_effort(records):
    """Prospective active-human minutes for one matched task and declared scope."""
    if not records:return dict(status='NOT_OBSERVED',ratio_fe_over_rdl=None,minutes={})
    ids=set();scopes=set();tasks=set();minutes={}
    for row in records:
        if not row.get('id') or row['id'] in ids:raise ValueError('Unique effort event IDs required')
        ids.add(row['id'])
        if row.get('arm') not in ('FE','RDL') or row.get('kind')!='human_active' or row.get('prospective') is not True:
            raise ValueError('Use prospective active human work for FE or RDL')
        x=row.get('minutes')
        if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<0:raise ValueError('Invalid minutes')
        if not row.get('scope') or not row.get('task'):raise ValueError('Declare task and work scope')
        scopes.add(row['scope']);tasks.add(row['task'])
        minutes[row['arm']]=minutes.get(row['arm'],0)+x
    if len(scopes)!=1 or len(tasks)!=1:raise ValueError('Compare the same task and work scope')
    status='INCOMPLETE' if len(minutes)!=2 else 'UNDEFINED_ZERO_DENOMINATOR' if minutes['RDL']==0 else 'OBSERVED'
    return dict(status=status,ratio_fe_over_rdl=minutes['FE']/minutes['RDL'] if status=='OBSERVED' else None,minutes=minutes)

def exit_gates(entries, failure_cases, review=None):
    """Conjunctive readiness checks; human rubric ratings must come from review."""
    seen=set();complete=matched=effort=temporal=0
    for row in entries:
        if not row.get('task') or row['task'] in seen:raise ValueError('One declared entry per task')
        seen.add(row['task'])
        if row.get('status') not in ('COMPLETE','INCOMPLETE','NOT_RUN') or row.get('split') not in ('val','test'):
            raise ValueError('Explicit experiment and split status required')
        if row.get('matched_fe') not in ('COMPLETE','INCOMPLETE','NOT_RUN') or row.get('effort') not in ('OBSERVED','NOT_OBSERVED','INCOMPLETE','UNDEFINED_ZERO_DENOMINATOR') or row.get('temporal') not in ('PASS','FAIL','NOT_ESTABLISHED','NOT_CHECKED'):
            raise ValueError('Unknown evidence status')
        c=row['status']=='COMPLETE' and row['split']=='test'
        f=c and row['matched_fe']=='COMPLETE'
        complete+=c;matched+=f;effort+=f and row['effort']=='OBSERVED';temporal+=c and row['temporal']=='PASS'
    if any(not isinstance(x,dict) or not x.get('evidence') or not x.get('limitation') for x in failure_cases):
        raise ValueError('Failure cases need evidence and a stated limitation')
    n=len(entries)
    checks=dict(task_coverage=n>=3 and complete==n,matched_fe=n>=3 and matched==n,
                human_effort=n>=3 and effort==n,temporal_audit=n>=3 and temporal==n,
                honest_failures=bool(failure_cases))
    defense='PENDING_WRITTEN_DEFENSE'
    if review is not None:
        scores=review.get('scores',[])
        if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip() or len(scores)!=5 or any(type(x) is not int or x not in (0,1,2) for x in scores):
            raise ValueError('Identified reviewer and five integer rubric scores (0–2) required')
        defense='PASS' if sum(scores)>=8 and min(scores)>0 else 'REVISION_REQUIRED'
    return dict(counts=dict(declared_tasks=n,completed_tasks=complete,matched_fe_tasks=matched,observed_effort_tasks=effort,temporal_pass_tasks=temporal),
                gates=checks,written_defense=defense,
                exit='INCOMPLETE' if not all(checks.values()) else defense,
                boundary='Machine checks verify declared evidence; reviewer must judge provenance, audit scope and prose.')

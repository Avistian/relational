"""Matched quality and prospectively observed marginal human effort.

Times are ISO8601 timestamps with explicit offsets. A 'complete' coverage flag is
an attestation by the logger, not proof that omitted human work did not occur.
"""
import math
from datetime import datetime

def paired_losses(queries, fe_predictions, rdl_predictions):
    """Align by the complete query key. Positive benefit favors RDL."""
    def keyed(rows,value):
        result={}
        for row in rows:
            key=(row['entity'],row['time']);v=float(row[value])
            if key in result or not math.isfinite(v):raise ValueError('Duplicate query or nonfinite value')
            result[key]=v
        return result
    truth=keyed(queries,'target');fe=keyed(fe_predictions,'prediction');rdl=keyed(rdl_predictions,'prediction')
    if not truth or set(truth)!=set(fe) or set(truth)!=set(rdl):raise ValueError('Full query populations must match')
    rows=[]
    for (entity,time),y in truth.items():
        f=abs(y-fe[(entity,time)]);g=abs(y-rdl[(entity,time)])
        rows.append(dict(entity=entity,time=time,fe_loss=f,rdl_loss=g,benefit=f-g))
    n=len(rows)
    return dict(n=n,fe_mae=sum(r['fe_loss'] for r in rows)/n,rdl_mae=sum(r['rdl_loss'] for r in rows)/n,
                benefit_mae=sum(r['benefit'] for r in rows)/n,rows=rows)

def summarize_effort(sessions, task, participant, assistance, coverage):
    """Count human marginal time separately from shared setup and machine time."""
    if not task or not participant or not assistance:raise ValueError('Declare task, participant and assistance')
    if set(coverage)!={'FE','RDL'} or any(v not in {'complete','partial','not_observed'} for v in coverage.values()):raise ValueError('Explicit method coverage required')
    marginal=dict(FE=0.,RDL=0.);shared=dict(FE=0.,RDL=0.);machine=dict(FE=0.,RDL=0.)
    seen=set();human_intervals=[];observed=set()
    for s in sessions:
        if not s.get('id') or s['id'] in seen:raise ValueError('Distinct session IDs required')
        seen.add(s['id'])
        if (s.get('task'),s.get('participant'),s.get('assistance'))!=(task,participant,assistance):raise ValueError('Mixed task, participant or assistance policy')
        method=s.get('method');kind=s.get('kind');scope=s.get('scope')
        if method not in marginal or kind not in {'human','machine'} or scope not in {'marginal','shared'} or not s.get('phase'):raise ValueError('Invalid session contract')
        try:
            start=datetime.fromisoformat(s['start']);end=datetime.fromisoformat(s['end'])
            if start.utcoffset() is None or end.utcoffset() is None or end<=start:raise ValueError('Closed positive interval with timezone required')
        except (KeyError,TypeError):raise ValueError('Closed timestamp interval required')
        hours=(end-start).total_seconds()/3600
        if kind=='human':
            if any(start<b and a<end for a,b in human_intervals):raise ValueError('Overlapping active human sessions')
            human_intervals.append((start,end))
            (marginal if scope=='marginal' else shared)[method]+=hours
            if scope=='marginal':observed.add(method)
        else:machine[method]+=hours
    for method,status in coverage.items():
        if status=='complete' and method not in observed:raise ValueError('Complete effort requires observed marginal human sessions')
        if status=='not_observed' and method in observed:raise ValueError('Observed sessions conflict with missingness')
        if method not in observed:marginal[method]=None
    return dict(task=task,participant=participant,assistance=assistance,coverage=dict(coverage),
                human_marginal_hours=marginal,human_shared_hours=shared,machine_hours=machine,
                interpretation='Descriptive logged effort; completeness is self-attested; order and experience are not controlled')

def effort_ratio(summary):
    """Ratio needs two complete comparable human logs and a positive denominator."""
    hours=summary['human_marginal_hours']
    for value in hours.values():
        if value is not None and (not math.isfinite(value) or value<0):raise ValueError('Finite nonnegative human hours required')
    if any(v!='complete' for v in summary['coverage'].values()):
        return dict(status='NOT_OBSERVED' if all(v=='not_observed' for v in summary['coverage'].values()) else 'INCOMPLETE',ratio=None,reduction_percent=None)
    fe=hours['FE'];rdl=hours['RDL']
    if fe is None or rdl is None:return dict(status='NOT_OBSERVED',ratio=None,reduction_percent=None)
    if rdl==0 or fe==0:return dict(status='UNDEFINED_ZERO_TIME',ratio=None,reduction_percent=None)
    return dict(status='OBSERVED_DESCRIPTIVE',ratio=fe/rdl,reduction_percent=100*(1-rdl/fe))

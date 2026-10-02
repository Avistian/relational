"""Visible, standard-library research-priority arithmetic; no model execution."""
import itertools,math
from decimal import Decimal


def admission(proposal):
    """Cost feasibility is separate from research priority and user authorization."""
    required={'preparation','training','selection','evaluation','retries','validation'}
    phases=proposal.get('phases_usd',{})
    for value in phases.values():
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
            raise ValueError('Costs must be finite nonnegative numbers or unknown')
    if proposal.get('source_audit')=='FAIL':return {'status':'BLOCKED','total_usd':None}
    if set(phases)!=required or any(v is None for v in phases.values()):return {'status':'NOT_ESTABLISHED','total_usd':None}
    total=sum((Decimal(str(v)) for v in phases.values()),Decimal(0))
    if total>Decimal('10'):return {'status':'OVER_CAP','total_usd':float(total)}
    if proposal.get('bound_verified') is not True or proposal.get('protocol_frozen') is not True or proposal.get('source_audit')!='PASS':
        return {'status':'NOT_ESTABLISHED','total_usd':float(total)}
    return {'status':'WITHIN_CAP','total_usd':float(total)}


def priority(impact,feasibility,weights):
    """Impact times weighted mean feasibility; subjective ordinal rubric, not probability."""
    if len(feasibility)!=3 or len(weights)!=3:raise ValueError('Exactly three feasibility dimensions')
    if any(type(v) is not int or not 1<=v<=5 for v in [impact]+list(feasibility)):raise ValueError('Scores must be integers 1..5')
    if any(type(v) is not int or v<=0 for v in weights):raise ValueError('Positive integer weights required')
    return impact*sum(s*w for s,w in zip(feasibility,weights))/sum(weights)


def sensitivity(cases):
    """All 27 weight triples; ties retained, ordered case IDs are not tie-break evidence."""
    ids=[c['id'] for c in cases]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Nonempty unique case IDs required')
    result=[]
    for weights in itertools.product([1,2,3],repeat=3):
        scores={c['id']:priority(c['impact'],c['feasibility'],weights) for c in cases}
        best=max(scores.values())
        result.append({'weights':list(weights),'scores':scores,'leaders':sorted(k for k,v in scores.items() if v==best)})
    return result

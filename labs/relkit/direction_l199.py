"""Visible contracts; authored rankings never authorize a research experiment."""
from fractions import Fraction


def priority(cases, weights):
    """Impact times weighted feasibility, with exact arithmetic and retained ties."""
    if len(weights)!=3 or any(type(w) is not int or w<=0 for w in weights):
        raise ValueError('Three positive integer weights required')
    ids=[c['id'] for c in cases]
    if not ids or any(not isinstance(i,str) or not i.strip() for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Nonempty unique candidate IDs required')
    scores={}
    for c in cases:
        values=[c['impact']]+c['feasibility']
        if len(values)!=4 or any(type(v) is not int or not 1<=v<=5 for v in values):
            raise ValueError('Impact and three feasibility ratings must be integers 1..5')
        scores[c['id']]=Fraction(c['impact']*sum(v*w for v,w in zip(c['feasibility'],weights)),sum(weights))
    best=max(scores.values())
    return dict(scores={k:float(v) for k,v in scores.items()},leaders=sorted(k for k,v in scores.items() if v==best))


def launch_gate(gates):
    """Every mandatory requirement must have evidence of PASS; unknown blocks."""
    required=['data','baseline','design','budget']
    if set(gates)!=set(required) or any(v not in ['PASS','FAIL','UNKNOWN'] for v in gates.values()):
        raise ValueError('Exactly four explicit PASS/FAIL/UNKNOWN requirements needed')
    blockers=[k for k in required if gates[k]!='PASS']
    return dict(state='DO_NOT_LAUNCH' if blockers else 'READY_FOR_REVIEW',blockers=blockers,authorization='NOT_GRANTED')


def memo_readiness(sections):
    """Check field presence only; author prose and form completion are not mastery."""
    required=['direction','hypothesis','contrast','baselines','metric','threshold','uncertainty','cost','stop','deferred','evidence','revision']
    if any(k not in sections or not isinstance(sections[k],str) for k in required):
        raise ValueError('Twelve textual memo fields required')
    missing=[k for k in required if not sections[k].strip()]
    return dict(state='DRAFT' if missing else 'READY_FOR_REVIEW',missing=missing,mastery='PENDING_WRITTEN_DEFENSE')

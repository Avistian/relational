"""Temporal evidence contracts. All times in a call share one declared unit."""
import math

def audit_observations(records):
    """Audit each dependency at its owner cutoff; schedules use publication time.

    An event must satisfy its declared event boundary AND be available.
    A schedule may describe a later event if already published. None is unknown.
    A timeless attribute uses kind='static' and event=None; availability still matters.
    """
    if not records:raise ValueError('No observations: no evidence')
    owners={};rows=[];counts={'PASS':0,'FAIL':0,'NOT_ESTABLISHED':0}
    for r in records:
        owner=r['owner'];t=r['cutoff'];event=r['event'];arrival=r['available']
        if r['rule'] not in ['strict','inclusive'] or r['kind'] not in ['event','schedule','static']:
            raise ValueError('Unknown rule or dependency kind')
        if not math.isfinite(t) or any(v is not None and not math.isfinite(v) for v in [event,arrival]):
            raise ValueError('Nonfinite timestamp')
        if owner in owners and owners[owner]!=t:raise ValueError('One owner must have one cutoff')
        owners[owner]=t;fail=[];unknown=[]
        if r['kind']=='event':
            if event is None:unknown.append('event time missing')
            elif event>t or (r['rule']=='strict' and event==t):fail.append('event boundary')
        if arrival is None:unknown.append('availability missing')
        elif arrival>t:fail.append('available after query')
        status='FAIL' if fail else 'NOT_ESTABLISHED' if unknown else 'PASS'
        counts[status]+=1;rows.append(dict(owner=owner,status=status,reasons=fail+unknown))
    status='FAIL' if counts['FAIL'] else 'NOT_ESTABLISHED' if counts['NOT_ESTABLISHED'] else 'PASS'
    return dict(status=status,counts=counts,rows=rows)

def audit_label_windows(queries, events, horizon):
    """Rebuild mean labels using (cutoff, cutoff+horizon], with complete keys."""
    if not queries or not math.isfinite(horizon) or horizon<=0:raise ValueError('Need queries and positive horizon')
    keys=set();by_entity={};rows=[];n=0
    for e in events:
        if not math.isfinite(e['time']) or not math.isfinite(e['value']):raise ValueError('Nonfinite event')
        by_entity.setdefault(e['entity'],[]).append(e)
    for q in queries:
        key=(q['entity'],q['time'])
        if key in keys:raise ValueError('Duplicate entity/cutoff')
        keys.add(key)
        if not math.isfinite(q['time']) or not math.isfinite(q['target']):raise ValueError('Nonfinite query')
        eligible=[e['value'] for e in by_entity.get(q['entity'],[]) if q['time']<e['time']<=q['time']+horizon]
        value=math.fsum(eligible)/len(eligible) if eligible else None;n+=len(eligible)
        passed=value is not None and math.isclose(value,q['target'],rel_tol=1e-12,abs_tol=1e-12)
        rows.append(dict(entity=q['entity'],time=q['time'],rebuilt=value,status='PASS' if passed else 'FAIL'))
    return dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',queries=len(rows),label_events=n,rows=rows)

def audit_verdict(checks, required):
    """A complete audit can still be inconclusive. Missing checks never pass."""
    if not required or len(set(required))!=len(required):raise ValueError('Unique nonempty required checks')
    by_id={}
    for c in checks:
        if c['id'] in by_id:raise ValueError('Duplicate check')
        if c['status'] not in ['PASS','FAIL','NOT_ESTABLISHED','NOT_CHECKED']:raise ValueError('Unknown evidence status')
        if not isinstance(c['evidence'],str) or not c['evidence'].strip():raise ValueError('Evidence or missing-evidence reason required')
        by_id[c['id']]=c
    missing=[k for k in required if k not in by_id]
    statuses=[c['status'] for c in by_id.values()]
    status=('FAIL' if 'FAIL' in statuses else 'NOT_CHECKED' if missing or 'NOT_CHECKED' in statuses
            else 'NOT_ESTABLISHED' if 'NOT_ESTABLISHED' in statuses else 'PASS')
    return dict(status=status,missing=missing,checked=len(set(required)&set(by_id)),required=len(required))

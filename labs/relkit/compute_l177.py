"""Visible compute-accounting contracts; no cloud execution."""

def reservation_total(attempts, overhead):
    """Keep every attempt reserved; do not add nested worker estimates again."""
    from decimal import Decimal
    import math
    def number(x):
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
            raise ValueError('A finite nonnegative measurement is required')
        return Decimal(str(x))
    total=number(overhead);seen=set()
    for a in attempts:
        if not isinstance(a['id'],str) or not a['id'] or a['id'] in seen:
            raise ValueError('Attempt identity must be unique and nonempty')
        seen.add(a['id'])
        total+=(number(a['seconds'])+number(a['lifecycle_seconds']))*number(a['rate'])
    return float(total)

def forecast_seconds(pilot_seconds, remaining_runs, margin, fixed_seconds):
    """Serial planning heuristic, not a statistical upper confidence bound."""
    import math
    values=list(pilot_seconds)
    if not values or type(remaining_runs) is not int or remaining_runs<0:
        raise ValueError('Need a nonempty pilot and an integer remaining count')
    for v in values+[margin,fixed_seconds]:
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
            raise ValueError('Invalid forecast input')
    if margin<1:raise ValueError('Margin must be at least one')
    return max(values)*remaining_runs*margin+fixed_seconds

def assess_plan(cost, wall_seconds, active_seconds, required_gib, available_gib,
                session_seconds, scientific_ok=True, planned_stop=8):
    """Conservative serial session: machine time plus active work, no overlap."""
    import math
    vals=dict(cost=cost,wall_seconds=wall_seconds,active_seconds=active_seconds,
              required_gib=required_gib,available_gib=available_gib,
              session_seconds=session_seconds,planned_stop=planned_stop)
    if type(scientific_ok) is not bool:raise ValueError('Scientific gate must be explicit')
    for v in vals.values():
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0):
            raise ValueError('Invalid measurement')
    if any(vals[k] is None for k in ['available_gib','session_seconds','planned_stop']):
        raise ValueError('Planning limits must be known')
    missing=[k for k in ['cost','wall_seconds','active_seconds','required_gib'] if vals[k] is None]
    session_total=None if wall_seconds is None or active_seconds is None else wall_seconds+active_seconds
    if not scientific_ok:status='SCIENTIFIC_STOP'
    elif cost is not None and cost>planned_stop:status='OVER_BUDGET'
    elif required_gib is not None and required_gib>available_gib:status='MEMORY_LIMIT'
    elif session_total is not None and session_total>session_seconds:status='EXCEEDS_SESSION'
    elif missing:status='INCOMPLETE_MEASUREMENT'
    else:status='FEASIBLE_SCENARIO'
    return dict(status=status,missing=missing,serial_session_seconds=session_total,
                meaning='Scenario admission only; not a measured future guarantee')

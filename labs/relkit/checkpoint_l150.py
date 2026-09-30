"""Checkpoint contracts. Numeric closeness never establishes current SOTA or mastery."""
import math,statistics

def select_candidate(rows):
    """Exactly three predeclared candidates; first-validation-minimum scores only."""
    if len(rows)!=3 or {r['lr'] for r in rows}!={.001,.003,.005}:
        raise ValueError('Require all three distinct planned learning rates')
    for r in rows:
        if r['split']!='val' or r['seed']!=100 or r['epochs']!=10 or not r['complete']:
            raise ValueError('Incomplete or wrong selection protocol')
        if any('test' in k.lower() for k in r):raise ValueError('Test information in selection packet')
        if not math.isfinite(r['selection_mae']) or r['selection_mae']<0:raise ValueError('Invalid validation score')
    return min(rows,key=lambda r:(r['selection_mae'],r['lr']))['lr']

def summarize_track(rows,track,expected_seeds):
    """No checkpoint inference, missing fit, duplicate seed or mixed track allowed."""
    seeds=[r['seed'] for r in rows]
    if len(set(expected_seeds))!=5 or len(seeds)!=5 or set(seeds)!=set(expected_seeds):
        raise ValueError('Require exactly the five planned seeds')
    if track not in ('reference','selected'):raise ValueError('Unknown primary training track')
    for r in rows:
        if r['track']!=track or not r['complete'] or r['epochs']!=10:raise ValueError('Not a complete primary fit')
        if not math.isfinite(r['test_mae']) or r['test_mae']<0:raise ValueError('Invalid test score')
    values=[r['test_mae'] for r in sorted(rows,key=lambda r:r['seed'])]
    return dict(track=track,seeds=sorted(seeds),values=values,mean=statistics.mean(values),sample_sd=statistics.stdev(values))

def checkpoint_verdict(protocol_ok,mean,competitive_audit,defense_passed):
    """Audit/defense booleans must come from independent review, never from the score."""
    if not math.isfinite(mean) or mean<0:raise ValueError('Invalid mean')
    return dict(protocol='PASS' if protocol_ok else 'FAIL',paper_score=('CLOSE' if abs(mean-3.798)<=.20+1e-12 else 'OUTSIDE_TOLERANCE') if protocol_ok else 'INCOMPARABLE',competitive='SUPPORTED_BY_SEPARATE_AUDIT' if competitive_audit and protocol_ok else 'NOT_ESTABLISHED',learner='DEFENDED' if defense_passed else 'PENDING_WRITTEN_DEFENSE')

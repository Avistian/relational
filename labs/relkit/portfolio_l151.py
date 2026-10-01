"""Checkpoint contracts. Numeric closeness never establishes current SOTA or mastery."""
import math,statistics

def select_candidate(rows):
    """Exactly three predeclared candidates; first-validation-maximum scores only."""
    if len(rows)!=3 or {r['lr'] for r in rows}!={.00005,.0001,.0002}:
        raise ValueError('Require all three distinct planned learning rates')
    for r in rows:
        if r['split']!='val' or r['seed']!=100 or r['epochs']!=20 or not r['complete']:
            raise ValueError('Incomplete or wrong selection protocol')
        if any('test' in k.lower() for k in r):raise ValueError('Test information in selection packet')
        if not math.isfinite(r['selection_auc']) or not 0<=r['selection_auc']<=1:raise ValueError('Invalid validation score')
    return min(rows,key=lambda r:(-r['selection_auc'],r['lr']))['lr']

def summarize_track(rows,track,expected_seeds):
    """No checkpoint inference, missing fit, duplicate seed or mixed track allowed."""
    seeds=[r['seed'] for r in rows]
    if len(set(expected_seeds))!=5 or len(seeds)!=5 or set(seeds)!=set(expected_seeds):
        raise ValueError('Require exactly the five planned seeds')
    if track not in ('reference','selected'):raise ValueError('Unknown primary training track')
    for r in rows:
        if r['track']!=track or not r['complete'] or r['epochs']!=20:raise ValueError('Not a complete primary fit')
        if not math.isfinite(r['test_auc']) or not 0<=r['test_auc']<=1:raise ValueError('Invalid test score')
    values=[r['test_auc'] for r in sorted(rows,key=lambda r:r['seed'])]
    return dict(track=track,seeds=sorted(seeds),values=values,mean=statistics.mean(values),sample_sd=statistics.stdev(values))

def portfolio_verdict(track,protocol_ok,mean,defense_passed):
    """A tuned course result cannot pass the released-protocol reproduction gate."""
    if track not in ('reference','selected') or not math.isfinite(mean) or not 0<=mean<=1:
        raise ValueError('Require primary track and AUROC in [0,1]')
    score='INCOMPARABLE'
    if protocol_ok and track=='reference':score='CLOSE' if abs(mean-.686)<=.01+1e-12 else 'OUTSIDE_TOLERANCE'
    return dict(protocol='PASS' if protocol_ok else 'FAIL',paper_score=score,historical_identity='NOT_ESTABLISHED',fresh_fe_comparison='NOT_RUN',learner='DEFENDED' if defense_passed else 'PENDING_WRITTEN_DEFENSE')

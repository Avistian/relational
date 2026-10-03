"""Learner-owned admission and validation contracts for L192."""
import math

def keyed_rows(rows):
    """Index integer entity/day records, rejecting duplicate complete query keys."""
    result={}
    for row in rows:
        if any(type(row.get(k)) is not int for k in ['entity','cutoff','label']):
            raise ValueError('Entity, cutoff day and binary label must be integers')
        if row['label'] not in [0,1]:raise ValueError('Binary label required')
        key=(row['entity'],row['cutoff'])
        if key in result:raise ValueError('Duplicate complete query key')
        result[key]=row
    return result

def available_history(history, query_cutoff):
    """Retain past query labels whose declared availability is no later than now."""
    if type(query_cutoff) is not int:raise ValueError('Integer cutoff day required')
    keyed_rows(history)
    for row in history:
        if type(row.get('available_at')) is not int or row['available_at']<row['cutoff']:
            raise ValueError('Missing or invalid label availability')
    return [r for r in history if r['cutoff']<query_cutoff and r['available_at']<=query_cutoff]

def select_candidate(records, expected):
    """Only a complete validation grid may select; ties follow the frozen order."""
    if not expected or len(set(expected))!=len(expected):raise ValueError('Invalid candidate grid')
    by_id={}
    for row in records:
        score=row.get('auc')
        if row.get('split')!='val' or type(score) not in [float,int] or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Finite validation AUROC in [0,1] required')
        if row.get('id') not in expected or row['id'] in by_id:raise ValueError('Unknown or duplicate candidate')
        by_id[row['id']]=score
    if set(by_id)!=set(expected):raise ValueError('Incomplete validation search')
    return max(expected,key=lambda name:by_id[name])

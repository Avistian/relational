"""Visible checkpoint arithmetic: saved evidence never authorizes a stronger claim."""
import itertools
import math
from fractions import Fraction


def keyed_auc(truth, predictions):
    """Rank AUROC with half credit for ties, aligned by (entity_id, cutoff)."""
    def index(rows):
        result={}
        for row in rows:
            if len(row)!=3:raise ValueError('Expected entity, cutoff, value')
            a,b,v=row;key=(a,b)
            if key in result or not math.isfinite(v):raise ValueError('Duplicate key or nonfinite value')
            result[key]=v
        return result
    labels=index(truth);probs=index(predictions)
    if not labels or labels.keys()!=probs.keys():raise ValueError('Incomplete query identity')
    if set(labels.values())!={0,1}:raise ValueError('Both binary classes required')
    if any(not 0<=v<=1 for v in probs.values()):raise ValueError('Probabilities outside [0,1]')
    ordered=sorted((probs[k],labels[k]) for k in labels)
    rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum+=((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
    npos=sum(labels.values());nneg=len(labels)-npos
    return (rank_sum-npos*(npos+1)/2)/(npos*nneg)


def claim_gate(kind, evidence):
    """Only the named saved metric is admissible from this replay's four checks."""
    if kind=='saved_metric':
        required=['authenticated','complete','metric_checked','comparable']
        return 'SUPPORTED_REPLAY' if all(evidence.get(k) is True for k in required) else 'BLOCKED'
    if kind=='fresh_training':return 'NOT_RUN'
    if kind in ['novelty','general_superiority']:return 'NOT_ESTABLISHED'
    raise ValueError('Unknown claim kind')


def rank_cases(cases):
    """Author-assigned ordinal rubric; exhaustive weight sensitivity, ties retained."""
    ids=[c['id'] for c in cases]
    if not ids or any(not isinstance(k,str) or not k for k in ids) or len(set(ids))!=len(ids):
        raise ValueError('Nonempty unique case IDs required')
    for c in cases:
        vals=[c['impact']]+c['feasibility']
        if len(vals)!=4 or any(type(v) is not int or not 1<=v<=5 for v in vals):
            raise ValueError('Impact and three feasibility scores must be integers 1..5')
    rows=[]
    for weights in itertools.product([1,2,3],repeat=3):
        scores={c['id']:Fraction(c['impact']*sum(a*b for a,b in zip(c['feasibility'],weights)),sum(weights)) for c in cases}
        best=max(scores.values())
        rows.append(dict(weights=list(weights),scores={k:float(v) for k,v in scores.items()},leaders=sorted(k for k,v in scores.items() if v==best)))
    return rows

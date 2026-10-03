"""Visible comparison contracts; matched declarations do not prove causal isolation."""
import numpy as np
FIELDS = ('task','query_keys','split','labels','features','support','preprocessing','visibility','selection','metric','budget_policy')

def compare_contracts(left, right):
    """Unknown values block evidence; any unequal known field blocks matching."""
    if not isinstance(left,dict) or not isinstance(right,dict):
        raise ValueError('Contracts must be mappings')
    if (set(left)|set(right))-set(FIELDS):
        raise ValueError('Unknown contract field')
    mismatches=[];unknown=[]
    for field in FIELDS:
        a,b=left.get(field),right.get(field)
        if any(v is not None and not isinstance(v,str) for v in (a,b)):
            raise ValueError('Contract values must be strings or null')
        if any(v is None or not v.strip() or v in ('UNKNOWN','NOT_ESTABLISHED') for v in (a,b)):
            unknown.append(field)
        elif a!=b:
            mismatches.append(field)
    return dict(status='INCOMPARABLE' if mismatches else 'NOT_ESTABLISHED' if unknown else 'MATCHED',mismatches=mismatches,unknown=unknown)

def paired_effect(left, right):
    """Join the ten support draws by identity; sample SD is conditional on this task."""
    maps=[]
    for records in (left,right):
        values={}
        for row in records:
            if not isinstance(row,dict) or set(row)!={'draw','auc'}:
                raise ValueError('Each row needs draw and auc')
            draw=row['draw'];auc=row['auc']
            if type(draw) is not int or draw not in range(10) or draw in values:
                raise ValueError('Invalid or duplicate draw')
            if isinstance(auc,bool) or not isinstance(auc,(int,float)) or not np.isfinite(auc) or not 0<=auc<=1:
                raise ValueError('Invalid AUROC')
            values[draw]=auc
        if set(values)!=set(range(10)):
            raise ValueError('All ten draws required')
        maps.append(values)
    deltas=np.array([maps[0][i]-maps[1][i] for i in range(10)])
    return dict(draws=list(range(10)),per_draw=deltas.tolist(),mean=float(deltas.mean()),sample_sd=float(deltas.std(ddof=1)),positive=int((deltas>0).sum()),uncertainty_unit='support draw on one fixed test population')

def claim_gate(contract_status, evidence, claim):
    """A descriptive replay is neither fresh inference nor an architecture intervention."""
    if contract_status not in ('MATCHED','INCOMPARABLE','NOT_ESTABLISHED') or evidence not in ('COMPLETE_REPLAY','INCOMPLETE'):
        raise ValueError('Invalid status')
    if claim not in ('selected_score','fresh_inference','architecture_cause','learner_mastery'):
        raise ValueError('Unknown claim')
    if claim=='learner_mastery':return 'PENDING_WRITTEN_DEFENSE'
    if claim=='fresh_inference':return 'NOT_RUN'
    if claim=='architecture_cause':return 'NOT_ESTABLISHED'
    if evidence!='COMPLETE_REPLAY':return 'INCOMPLETE'
    return 'SUPPORTED_DESCRIPTIVE_REPLAY' if contract_status=='MATCHED' else contract_status

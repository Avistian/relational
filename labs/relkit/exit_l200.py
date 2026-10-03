"""Visible exam contracts; evidence completeness does not establish learner mastery."""
import numpy as np

def keyed_auc(keys, labels, prediction_keys, probabilities):
    """Join on complete entity/cutoff keys; AUROC uses half credit for ties."""
    k=np.asarray(keys);pk=np.asarray(prediction_keys);y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if k.ndim!=2 or k.shape[1]!=2 or pk.shape!=k.shape or y.shape!=(len(k),) or p.shape!=y.shape:
        raise ValueError('Invalid shapes')
    a=list(map(tuple,k.tolist()));b=list(map(tuple,pk.tolist()))
    if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)!=set(b):raise ValueError('Missing or duplicate full key')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or set(y.tolist())!={0,1}:raise ValueError('Invalid label/probability')
    lookup=dict(zip(b,p));ordered=np.array([lookup[key] for key in a]);pos=ordered[y==1];neg=ordered[y==0]
    return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))

def complete_grid(records):
    """Require exactly 3 configurations × 10 draws, never summarize a partial grid."""
    expected={(a,s) for a in ['RDBPFN','RDBPFN_single','TabICLv1.1'] for s in range(10)};seen=set()
    for row in records:
        if not isinstance(row,dict) or any(k not in row for k in ['arm','seed','rows','support']):raise ValueError('Missing fields')
        if not isinstance(row['arm'],str) or any(type(row[k]) is not int for k in ['seed','rows','support']):raise ValueError('Invalid field type')
        identity=(row['arm'],row['seed'])
        if identity not in expected or identity in seen or row['rows']!=702 or row['support']!=512:raise ValueError('Wrong run grid')
        seen.add(identity)
    if seen!=expected:raise ValueError('Incomplete grid')
    return dict(runs=len(seen),predictions=702*len(seen))

def exit_gate(reproduction, proposal, defense):
    """Author verification cannot fill the learner's proposal or oral/written defense."""
    values=dict(reproduction=reproduction,proposal=proposal,defense=defense)
    if any(v not in ['PASS','FAIL','PENDING'] for v in values.values()):raise ValueError('Unknown state')
    blockers=[k for k,v in values.items() if v!='PASS']
    return dict(state='INCOMPLETE' if blockers else 'READY_FOR_TEACHER_REVIEW',blockers=blockers)

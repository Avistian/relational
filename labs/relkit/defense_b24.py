"""Three live learner mechanisms for the B24 research defense."""
import math
import numpy as np

def paired_effect(records, reference, challenger):
    """Pair all declared draws by identity; positive means higher challenger AUROC."""
    if not isinstance(reference,str) or not isinstance(challenger,str) or reference==challenger:
        raise ValueError('Choose two distinct arms')
    by_arm={reference:{},challenger:{}}
    for row in records:
        if not isinstance(row,dict) or not {'arm','seed','auc'}<=row.keys():raise ValueError('Missing record fields')
        if row['arm'] not in by_arm:continue
        seed=row['seed'];value=row['auc']
        if type(seed) is not int or seed<0 or seed in by_arm[row['arm']]:raise ValueError('Invalid or duplicate draw')
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid AUROC')
        by_arm[row['arm']][seed]=value
    seeds=sorted(by_arm[reference])
    if len(seeds)<2 or set(seeds)!=set(by_arm[challenger]):raise ValueError('Unpaired draws')
    delta=np.array([by_arm[challenger][s]-by_arm[reference][s] for s in seeds])
    return dict(reference=reference,challenger=challenger,seeds=seeds,per_seed=delta.tolist(),mean=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),positive=int((delta>0).sum()))

def defense_gate(scores, reproduction, leakage, assessed):
    """Hypothetical eligibility; human assessment is a separate explicit input."""
    axes={'protocol','baselines','reproducibility','interpretation','falsifiability'}
    if not isinstance(scores,dict) or set(scores)!=axes or any(type(v) is not int or not 0<=v<=2 for v in scores.values()):raise ValueError('Five integer scores from 0 to 2 required')
    if any(type(v) is not bool for v in [reproduction,leakage,assessed]):raise ValueError('Explicit Boolean evidence flags required')
    total=sum(scores.values());blockers=[]
    if total<8:blockers.append('SCORE_BELOW_8')
    if 0 in scores.values():blockers.append('ZERO_AXIS')
    if not reproduction:blockers.append('REPRODUCTION_PENDING')
    if leakage:blockers.append('UNRESOLVED_LEAKAGE')
    eligible=not blockers
    learner=('PASS' if eligible else 'REVISION_REQUIRED') if assessed else 'PENDING_WRITTEN_DEFENSE'
    return dict(total=total,eligible=eligible,blockers=blockers,learner=learner)

def falsification_contract(tests, used_tasks):
    """Validate two proposed tests; readiness never implies they were executed."""
    if not isinstance(tests,list) or len(tests)!=2:raise ValueError('Exactly two falsification tests required')
    kinds=[]
    for t in tests:
        if not isinstance(t,dict):raise ValueError('Each test is a record')
        for key in ['kind','task','challenger','reference','metric','orientation','action']:
            if not isinstance(t.get(key),str) or not t[key].strip():raise ValueError('Missing '+key)
        if t['kind'] not in ['simpler_baseline','untouched_task']:raise ValueError('Invalid test kind')
        if t['orientation'] not in ['higher','lower']:raise ValueError('Define the favorable metric direction')
        if t['challenger']==t['reference']:raise ValueError('Distinct comparators required')
        threshold=t.get('threshold')
        if type(threshold) not in [int,float] or not math.isfinite(threshold):raise ValueError('Finite numeric threshold required')
        if t.get('information_matched') is not True or t.get('budget_matched') is not True:raise ValueError('Match information and selection budgets')
        if t['kind']=='untouched_task' and t['task'] in used_tasks:raise ValueError('This task has already informed development')
        kinds.append(t['kind'])
    if set(kinds)!={'simpler_baseline','untouched_task'}:raise ValueError('Both kinds required')
    return dict(ready=True,tests=2,execution='NOT_RUN')

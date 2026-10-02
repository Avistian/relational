"""Live learner contracts for matched relational-model evaluation."""
def keyed_auc(keys, labels, prediction_keys, probabilities):
    """Align full entity/cutoff keys; average ties in the pairwise AUROC."""
    import numpy as np
    q=[tuple(k) for k in keys];p=[tuple(k) for k in prediction_keys]
    y=np.asarray(labels);v=np.asarray(probabilities,dtype=float)
    if not q or any(len(k)!=2 for k in q+p) or len(set(q))!=len(q) or len(set(p))!=len(p) or set(q)!=set(p):
        raise ValueError('Require identical unique full query-key populations')
    if y.shape!=(len(q),) or v.shape!=(len(p),) or set(y)!={0,1} or not np.isfinite(v).all() or (v<0).any() or (v>1).any():
        raise ValueError('Require binary labels and finite probabilities')
    lookup=dict(zip(p,v));a=np.array([lookup[k] for k in q]);neg=np.sort(a[y==0]);pos=a[y==1]
    return float(np.mean((np.searchsorted(neg,pos,'left')+np.searchsorted(neg,pos,'right'))/(2*len(neg))))

def comparison_gate(contracts):
    """An affordable experiment still needs matching information and passed audits."""
    import math
    arms=['RDBPFN','RelGNN','RDBLearn'];reasons=[]
    if set(contracts)!=set(arms):return dict(status='INCOMPLETE_CONTRACT',reasons=['Require all three named arms'])
    fields=['task','horizon_days','positive','query_digest','database_digest','support_digest','cost_ceiling']
    for field in fields:
        values=[contracts[a].get(field) for a in arms]
        if any(v is None or v=='' for v in values) or any(v!=values[0] for v in values):reasons.append('Unmatched '+field)
    for arm in arms:
        c=contracts[arm]
        for field,want in [('target_history','NONE'),('temporal_audit','PASS'),('preprocessing_audit','PASS'),('selection','VALIDATION_ONLY')]:
            if c.get(field)!=want:reasons.append(arm+': '+field+'='+str(c.get(field)))
        cost=c.get('cost_ceiling')
        if type(cost) not in (int,float) or not math.isfinite(cost) or not 0<cost<=1.5:reasons.append(arm+': invalid cost ceiling')
    health=contracts['RelGNN'].get('training_health')
    if health!='PASS':reasons.append('RelGNN: training_health='+str(health))
    status='INCOMPLETE_TRAINING_HEALTH_GATE' if str(health).startswith('FAIL') else ('INCOMPLETE_COMPARABILITY_GATE' if reasons else 'READY_FOR_PILOT')
    return dict(status=status,reasons=reasons)

def select_validation(records, expected_configs, seeds, epochs):
    """Choose configuration by mean best validation AUROC; earliest epoch breaks ties."""
    import math
    expected={(c,s,e) for c in expected_configs for s in seeds for e in range(1,epochs+1)};seen=set();best={}
    if not expected or len(set(expected_configs))!=len(expected_configs) or len(set(seeds))!=len(seeds):raise ValueError('Invalid expected grid')
    for row in records:
        key=(row['config'],row['seed'],row['epoch']);auc=row['validation_auc']
        if key in seen or key not in expected or type(auc) not in (int,float) or not math.isfinite(auc) or not 0<=auc<=1:raise ValueError('Invalid tuning record')
        seen.add(key);pair=key[:2];candidate=(-auc,row['epoch'])
        if pair not in best or candidate<best[pair]:best[pair]=candidate
    if seen!=expected:raise ValueError('Incomplete tuning grid')
    ranking=[(-math.fsum(-best[c,s][0] for s in seeds)/len(seeds),i,c) for i,c in enumerate(expected_configs)]
    negative,_,chosen=min(ranking)
    return dict(config=chosen,epochs={str(s):best[chosen,s][1] for s in seeds},mean_validation_auc=-negative)

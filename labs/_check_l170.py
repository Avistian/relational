"""Meaningful live contracts: pairing, target labels, and missing evidence."""
def check170(pair_fn, adaptation_fn, gate_fn):
    import math
    records=[]
    for db in ['rel-f1','rel-trial']:
        for k in [64,128,256,512,1024]:
            for s in range(10):
                for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1']:
                    # Opposing task effects; seed effects expose incorrect pairing.
                    base=.5+.01*s
                    delta=(.02 if db=='rel-f1' else -.04)+.002*s
                    value=base+(delta if arm=='RDBPFN' else -.1 if arm=='RDBPFN_single' else 0)
                    records.append(dict(database=db,context=k,seed=s,arm=arm,auc=value))
    result=pair_fn(records[::-1])
    assert len(result['by_task_context'])==10
    a=result['by_task_context']['rel-f1/512'];b=result['by_task_context']['rel-trial/512']
    assert abs(a['mean']-.029)<1e-12 and abs(b['mean']+.031)<1e-12
    assert a['positive_seeds']==10 and b['positive_seeds']==0
    assert abs(a['sample_sd']-math.sqrt(sum((.002*s-.009)**2 for s in range(10))/9))<1e-12
    assert abs(result['macro_by_context']['512']+.001)<1e-12
    assert result['uncertainty_unit']=='SUPPORT_DRAWS_WITHIN_FIXED_TASK'
    for bad in [records[:-1],records+[records[0]], [dict(r,auc=float('nan')) if i==0 else r for i,r in enumerate(records)], [dict(r,seed=True) if i==0 else r for i,r in enumerate(records)]]:
        try:pair_fn(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid or incomplete grid accepted')
    assert adaptation_fn(512,0)=='LABELED_CONTEXT'
    assert adaptation_fn(0,0)=='NO_TARGET_LABELS_OR_UPDATES'
    assert adaptation_fn(512,1)=='TARGET_WEIGHT_UPDATES'
    for args in [(0,1),(-1,0),(True,0),(1,-1),(1,1.5)]:
        try:adaptation_fn(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid adaptation declaration accepted')
    keys=['artifacts','metrics','dfs','temporal','exposure','fresh_training','matched_pipeline','heldout_databases','repeatability']
    facts={k:False for k in keys};facts.update(artifacts=True,metrics=True)
    assert gate_fn('saved_replay',facts)==dict(status='READY_FOR_REVIEW',missing=[])
    assert gate_fn('full_pipeline',facts)==dict(status='NOT_ESTABLISHED',missing=['dfs','temporal'])
    assert gate_fn('fresh_pretraining',facts)['missing']==['fresh_training','exposure']
    assert gate_fn('general_advantage',facts)['missing']==['matched_pipeline','heldout_databases','temporal','exposure']
    assert gate_fn('exact_repeatability',facts)['missing']==['repeatability']
    assert gate_fn('general_advantage',{k:True for k in keys})['status']=='READY_FOR_REVIEW'
    for claim,ev in [('unknown',facts),('saved_replay',{}),('saved_replay',dict(facts,dfs='unknown'))]:
        try:gate_fn(claim,ev)
        except ValueError:pass
        else:raise AssertionError('Invalid evidence declaration accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.design_l170 import paired_design,adaptation_mode,claim_gate
    print(check170(paired_design,adaptation_mode,claim_gate))

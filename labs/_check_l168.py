"""Hand-worked feedback, including traps that change the scientific conclusion."""
def check168(regime_fn, paired_fn, macro_fn):
    import copy
    import numpy as np
    assert regime_fn(['source'],'target',True,512,0)=={'database_holdout':'HELD_OUT','adaptation':'FEW_SHOT_ICL'}
    assert regime_fn(['target'],'target',True,512,0)['database_holdout']=='SEEN'
    assert regime_fn(iter(['target']),'target',True,512,0)['database_holdout']=='SEEN'
    assert regime_fn(['source'],'target',False,0,0)=={'database_holdout':'NOT_ESTABLISHED','adaptation':'ZERO_LABEL_ZERO_GRADIENT'}
    assert regime_fn(['source'],'target',True,512,5)['adaptation']=='SUPERVISED_ADAPTATION'
    rows=[]
    # Reversed seed order defeats positional pairing. Different test sizes must not weight the macro.
    for db,base,delta,n in [('small',.6,.1,2),('large',.8,-.2,200)]:
        for seed in [1,0]:
            for arm in ['RDBPFN','TabICLv1.1']:
                rows.append(dict(database=db,seed=seed,arm=arm,auc=base+seed*.01+(delta if arm=='RDBPFN' else 0),test_rows=n,metric='AUROC'))
    result=paired_fn(rows,['small','large'],[0,1])
    np.testing.assert_allclose(result['small']['per_seed'],[.1,.1])
    np.testing.assert_allclose(result['large']['per_seed'],[-.2,-.2])
    m=macro_fn(result)
    assert m['databases']==2 and abs(m['macro_gain']-(-.05))<1e-12
    assert m['positive_databases']==1
    for bad in [rows[:-1],rows+[rows[0]]]:
        try:paired_fn(bad,['small','large'],[0,1])
        except ValueError:pass
        else:raise AssertionError('Missing or duplicate run accepted')
    bad=copy.deepcopy(rows);bad[0]['metric']='accuracy'
    try:paired_fn(bad,['small','large'],[0,1])
    except ValueError:pass
    else:raise AssertionError('Mixed metric accepted')
    bad=copy.deepcopy(rows);bad[0]['test_rows']=3
    try:paired_fn(bad,['small','large'],[0,1])
    except ValueError:pass
    else:raise AssertionError('Unmatched populations accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.generalization_l168 import transfer_regime,paired_gains,database_macro
    print(check168(transfer_regime,paired_gains,database_macro))

"""Behavioral checks for identity, information access and validation-only selection."""
def checks(score_fn, gate_fn, select_fn):
    import copy,math
    keys=[[1,10],[1,20],[2,10],[3,10]]; y=[0,1,1,0]
    assert score_fn(keys,y,keys[::-1],[.2,.8,.9,.1])==1
    assert score_fn(keys,y,keys,[.5]*4)==.5
    for pk,p in [(keys[:-1],[.1,.2,.3]),([keys[0]]*4,[.1]*4),(keys,[0,1,float('nan'),.2]),(keys,[0,1,2,.2])]:
        try:score_fn(keys,y,pk,p)
        except ValueError:pass
        else:raise AssertionError('Invalid identity/probability accepted')
    base=dict(task='driver-dnf',horizon_days=30,positive='DNF',query_digest='q',database_digest='d',support_digest='s',target_history='NONE',temporal_audit='PASS',preprocessing_audit='PASS',selection='VALIDATION_ONLY',cost_ceiling=1.5,training_health='PASS')
    contracts={a:copy.deepcopy(base) for a in ['RDBPFN','RelGNN','RDBLearn']}
    assert gate_fn(contracts)['status']=='READY_FOR_PILOT'
    for field,value in [('horizon_days',60),('positive','FINISH'),('query_digest','other'),('database_digest','other'),('support_digest','other'),('target_history','ALL_TRAIN'),('temporal_audit','FAIL'),('preprocessing_audit','NOT_CHECKED'),('selection','TEST'),('cost_ceiling',2)]:
        bad=copy.deepcopy(contracts);bad['RDBLearn'][field]=value
        assert gate_fn(bad)['status']!='READY_FOR_PILOT',field
    bad=copy.deepcopy(contracts);bad['RelGNN']['training_health']='FAIL_NONFINITE_GRADIENT'
    assert gate_fn(bad)['status']=='INCOMPLETE_TRAINING_HEALTH_GATE'
    assert gate_fn({'RDBPFN':base})['status']!='READY_FOR_PILOT'
    rows=[dict(config=c,seed=s,epoch=e,validation_auc=(.8 if c=='a' else .7),test_auc=(.1 if c=='a' else .99)) for c in ['a','b'] for s in [0,1,2] for e in [1,2]]
    selected=select_fn(rows,['a','b'],[0,1,2],2)
    assert selected==dict(config='a',epochs={'0':1,'1':1,'2':1},mean_validation_auc=math.fsum([.8]*3)/3)
    for r in rows:r['test_auc']=1-r['test_auc']
    assert select_fn(rows,['a','b'],[0,1,2],2)==selected
    for bad in [rows[:-1],rows+[rows[0]]]:
        try:select_fn(bad,['a','b'],[0,1,2],2)
        except ValueError:pass
        else:raise AssertionError('Incomplete tuning grid accepted')
    return 'PASS'
if __name__=='__main__':
    from relkit.comparison_l178 import keyed_auc,comparison_gate,select_validation
    print(checks(keyed_auc,comparison_gate,select_validation))

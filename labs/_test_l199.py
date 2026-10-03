"""Behavioral contracts: rankings cannot override missing mandatory evidence."""
def checks(priority, launch_gate, memo_readiness):
    import itertools
    def rejects(fn,*args):
        try: fn(*args)
        except ValueError: return
        raise AssertionError('Invalid decision input accepted')
    cases=[dict(id='a',impact=4,feasibility=[5,4,5]),dict(id='b',impact=5,feasibility=[3,2,4]),dict(id='c',impact=4,feasibility=[3,3,2])]
    r=priority(cases,[1,1,1]);assert r['scores']==dict(a=56/3,b=15.,c=32/3) and r['leaders']==['a']
    tied=[dict(id='b',impact=4,feasibility=[5,4,5]),cases[0]]
    assert priority(tied,[1,2,3])['leaders']==['a','b']
    assert priority(cases,[2,2,2])==r
    changed=[dict(c) for c in cases];changed[0]['impact']=3
    assert priority(changed,[1,1,1])['leaders']==['b']
    for w in [[0,1,1],[True,1,1],[1,2],[-1,2,3],[float('nan'),1,1]]:rejects(priority,cases,w)
    rejects(priority,[],[1,1,1]);rejects(priority,cases+[cases[0]],[1,1,1])
    rejects(priority,[dict(id='a',impact=6,feasibility=[5,4,5])],[1,1,1])
    rejects(priority,[dict(id='a',impact=True,feasibility=[5,4,5])],[1,1,1])
    keys=['data','baseline','design','budget']
    for values in itertools.product(['PASS','FAIL','UNKNOWN'],repeat=4):
        gates=dict(zip(keys,values));r=launch_gate(gates)
        assert r['state']==('READY_FOR_REVIEW' if all(x=='PASS' for x in values) else 'DO_NOT_LAUNCH')
        assert r['blockers']==[k for k in keys if gates[k]!='PASS']
        assert r['authorization']=='NOT_GRANTED'
    rejects(launch_gate,{'data':'PASS'});rejects(launch_gate,dict.fromkeys(keys,True))
    rejects(launch_gate,dict(dict.fromkeys(keys,'PASS'),novelty='PASS'))
    fields=['direction','hypothesis','contrast','baselines','metric','threshold','uncertainty','cost','stop','deferred','evidence','revision']
    blank=dict.fromkeys(fields,' ');r=memo_readiness(blank)
    assert r==dict(state='DRAFT',missing=fields,mastery='PENDING_WRITTEN_DEFENSE')
    filled=dict.fromkeys(fields,'Needs human review.')
    assert memo_readiness(filled)==dict(state='READY_FOR_REVIEW',missing=[],mastery='PENDING_WRITTEN_DEFENSE')
    rejects(memo_readiness,{});rejects(memo_readiness,dict(filled,cost=None))
    return dict(status='PASS',gate_states=81,contracts=3)
if __name__=='__main__':
    from relkit.direction_l199 import priority,launch_gate,memo_readiness
    print(checks(priority,launch_gate,memo_readiness))

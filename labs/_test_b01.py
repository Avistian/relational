"""Behavioral checks; notebooks supply their own implementations."""
import copy

def checks(compare, paired, gate):
    fields=('task','query_keys','split','labels','features','support','preprocessing','visibility','selection','metric','budget_policy')
    a={k:'frozen-'+k for k in fields};b=dict(a)
    assert compare(a,b)=={'status':'MATCHED','mismatches':[],'unknown':[]}
    for k in fields:
        b=dict(a);b[k]='different'
        assert compare(a,b)=={'status':'INCOMPARABLE','mismatches':[k],'unknown':[]}
        for value in [None,'','UNKNOWN','NOT_ESTABLISHED']:
            b=dict(a);b[k]=value
            assert compare(a,b)['status']=='NOT_ESTABLISHED'
            assert compare(a,b)['unknown']==[k]
    b=dict(a);del b['task'];assert compare(a,b)['status']=='NOT_ESTABLISHED'
    b=dict(a);b['features']='different';b['visibility']='UNKNOWN'
    assert compare(a,b)['status']=='INCOMPARABLE' and compare(a,b)['unknown']==['visibility']
    def rejects(fn,*args):
        try:fn(*args)
        except ValueError:return
        raise AssertionError('Bad input accepted')
    b=dict(a);b['typo']=1;rejects(compare,a,b)
    b=dict(a);b['task']=['same'];rejects(compare,a,b)
    l=[{'draw':i,'auc':.7+i*.001} for i in range(10)]
    r=[{'draw':i,'auc':.6+i*.001} for i in reversed(range(10))]
    x=paired(l,r);assert x['draws']==list(range(10)) and abs(x['mean']-.1)<1e-12 and x['positive']==10
    rejects(paired,l[:-1],r);rejects(paired,l+[l[0]],r)
    for v in [float('nan'),-0.1,1.1]:
        q=copy.deepcopy(l);q[0]['auc']=v;rejects(paired,q,r)
    q=copy.deepcopy(l);q[0]['draw']=False;rejects(paired,q,r)
    assert gate('MATCHED','COMPLETE_REPLAY','selected_score')=='SUPPORTED_DESCRIPTIVE_REPLAY'
    assert gate('MATCHED','COMPLETE_REPLAY','fresh_inference')=='NOT_RUN'
    assert gate('MATCHED','COMPLETE_REPLAY','architecture_cause')=='NOT_ESTABLISHED'
    assert gate('MATCHED','COMPLETE_REPLAY','learner_mastery')=='PENDING_WRITTEN_DEFENSE'
    assert gate('INCOMPARABLE','COMPLETE_REPLAY','selected_score')=='INCOMPARABLE'
    assert gate('NOT_ESTABLISHED','COMPLETE_REPLAY','selected_score')=='NOT_ESTABLISHED'
    assert gate('MATCHED','INCOMPLETE','selected_score')=='INCOMPLETE'
    rejects(gate,'MATCHED','COMPLETE_REPLAY','universal_winner')
    return 'PASS'

if __name__=='__main__':
    from relkit.comparison_b01 import compare_contracts,paired_effect,claim_gate
    print(checks(compare_contracts,paired_effect,claim_gate))

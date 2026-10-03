"""Behavioral contracts for a checkpoint that cannot promote incomplete evidence."""
import copy

def checks(score,gate,rank):
    truth=[(1,10,0),(1,20,1),(2,10,1),(2,20,0)]
    pred=[(2,20,.5),(1,20,.9),(1,10,.1),(2,10,.5)]
    assert score(truth,pred)==.875
    assert score(truth,[(a,b,.5) for a,b,_ in truth])==.5
    for bad in [pred[:-1],pred+[pred[0]],[(9,9,.1)]+pred[1:],[(a,b,float('nan')) for a,b,_ in pred]]:
        try:score(truth,bad)
        except ValueError:pass
        else:raise AssertionError('Bad predictions accepted')
    base=dict(authenticated=True,complete=True,metric_checked=True,comparable=True,novelty_reviewed=False)
    assert gate('saved_metric',base)=='SUPPORTED_REPLAY'
    for key in ['authenticated','complete','metric_checked','comparable']:
        assert gate('saved_metric',dict(base,**{key:False}))=='BLOCKED'
    assert gate('novelty',base)=='NOT_ESTABLISHED'
    assert gate('fresh_training',base)=='NOT_RUN'
    assert gate('general_superiority',base)=='NOT_ESTABLISHED'
    cases=[dict(id='a',impact=4,feasibility=[5,4,5]),dict(id='b',impact=5,feasibility=[3,2,4]),dict(id='c',impact=4,feasibility=[3,3,2])]
    rows=rank(cases);assert len(rows)==27
    assert rows[0]['scores']=={'a':56/3,'b':15.,'c':32/3}
    assert all(r['leaders']==['a'] for r in rows)
    tied=rank([cases[0],dict(cases[0],id='other')]);assert all(r['leaders']==['a','other'] for r in tied)
    for bad in [[],[cases[0],cases[0]],[dict(cases[0],impact=0)],[dict(cases[0],feasibility=[1,2])]]:
        try:rank(bad)
        except ValueError:pass
        else:raise AssertionError('Malformed ranking accepted')
    return 'PASS'
if __name__=='__main__':
    from relkit.checkpoint_l190 import keyed_auc,claim_gate,rank_cases
    print(checks(keyed_auc,claim_gate,rank_cases))

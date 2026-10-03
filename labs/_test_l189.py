"""Behavioral tests shared by the student lab and author validation."""
from copy import deepcopy

def checks(admission, priority, sensitivity):
    complete={'phases_usd':{'preparation':1,'training':4,'selection':1,'evaluation':1,'retries':1,'validation':1},'bound_verified':True,'protocol_frozen':True,'source_audit':'PASS'}
    assert admission(complete)=={'status':'WITHIN_CAP','total_usd':9.0}
    for key in ['training','retries']:
        x=deepcopy(complete);x['phases_usd'][key]=None
        assert admission(x)['status']=='NOT_ESTABLISHED'
    x=deepcopy(complete);x['phases_usd'].pop('retries');assert admission(x)['status']=='NOT_ESTABLISHED'
    x=deepcopy(complete);x['phases_usd']['training']=5;assert admission(x)['status']=='WITHIN_CAP'
    x['phases_usd']['training']=5.01;assert admission(x)['status']=='OVER_CAP'
    x=deepcopy(complete);x['source_audit']='FAIL';assert admission(x)['status']=='BLOCKED'
    x=deepcopy(complete);x['phases_usd']['training']=-1
    try:admission(x)
    except ValueError:pass
    else:raise AssertionError('Negative cost accepted')
    assert priority(4,[5,4,5],[1,1,1])==56/3
    assert priority(5,[4,3,2],[1,1,1])==15
    for args in [(True,[5,4,5],[1,1,1]),(4,[0,4,5],[1,1,1]),(4,[5,4],[1,1,1]),(4,[5,4,5],[1,0,1])]:
        try:priority(*args)
        except ValueError:pass
        else:raise AssertionError('Malformed rubric accepted')
    tied=[dict(id='a',impact=4,feasibility=[3,3,3]),dict(id='b',impact=3,feasibility=[4,4,4])]
    grid=sensitivity(tied)
    assert len(grid)==27 and all(r['leaders']==['a','b'] for r in grid)
    switched=sensitivity([dict(id='a',impact=4,feasibility=[5,1,1]),dict(id='b',impact=4,feasibility=[1,5,1])])
    assert any(r['leaders']==['a'] for r in switched) and any(r['leaders']==['b'] for r in switched)
    try:sensitivity(tied+tied)
    except ValueError:pass
    else:raise AssertionError('Duplicate case identities accepted')
    return 'PASS'

if __name__=='__main__':
    from relkit.gaps_l189 import admission,priority,sensitivity
    print(checks(admission,priority,sensitivity))

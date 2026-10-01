"""Behavioral contracts: evidence lineage, bounded claims and falsifiability."""
from relkit.synthesis_l158 import evidence_coverage, claim_verdict, validate_falsifier

def rejects(fn,*args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError('Invalid evidence was accepted')

def check(coverage=evidence_coverage,verdict=claim_verdict,falsifier=validate_falsifier):
    rows=[dict(id='a',task='f1/position',database='f1',prediction_hash='abc',status='COMPLETE',split='test'),
          dict(id='b',task='f1/position',database='f1',prediction_hash='abc',status='COMPLETE',split='test'),
          dict(id='c',task='trial/rank',database='trial',prediction_hash=None,status='INCOMPLETE',split='test')]
    r=coverage(rows)
    assert r['completed_tasks']==1 and r['completed_databases']==1 and r['unique_prediction_packets']==1
    assert r['reused_packets']==[['a','b']] and r['missing_tasks']==['trial/rank']
    rejects(coverage,rows+[rows[0]])
    rejects(coverage,[dict(rows[0],split='val')])
    rejects(coverage,[dict(rows[0],prediction_hash=None)])
    e=dict(matched_tasks=1,required_tasks=3,benefit=-.06,interval=[-.31,.17],human_effort='NOT_OBSERVED',historical_availability='NOT_ESTABLISHED')
    assert verdict('local_quality',e)=='POINT_ESTIMATE_FAVORS_FE; SUPERIORITY_NOT_ESTABLISHED'
    assert verdict('portfolio_superiority',e)=='NOT_ESTABLISHED'
    assert verdict('human_effort',e)=='NOT_OBSERVED'
    assert verdict('leak_free',e)=='NOT_ESTABLISHED'
    rejects(verdict,'unknown',e)
    rejects(verdict,'local_quality',dict(e,interval=[.2,-.2]))
    rejects(verdict,'local_quality',dict(e,benefit=float('nan')))
    f=dict(task='new-db/task',metric='MAE',direction='lower',minimum_benefit=.1,decision_split='test',selection_split='val',baseline='SQL + LightGBM',information_policy='same owner cutoff',disconfirm_if='benefit upper interval bound below threshold')
    assert falsifier(f)['status']=='SPECIFIED_NOT_EXECUTED'
    rejects(falsifier,dict(f,selection_split='test'))
    rejects(falsifier,dict(f,minimum_benefit=-1))
    rejects(falsifier,dict(f,baseline=''))
    print('PASS: three learner contracts; malformed and unsupported claims rejected')

if __name__=='__main__':check()

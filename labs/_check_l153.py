"""Meaningful contracts: multi-positive AP, query ownership, temporal/collision separation."""
import copy, math
from relkit.recommendation_l153 import ranking_metrics,negative_audit,portfolio_entry

def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected invalid evidence to be rejected')

def check_ranking(fn):
    truth=[{'entity':1,'time':10,'positives':[1,3,4]}, {'entity':1,'time':20,'positives':[2]}]
    pred=[{'entity':1,'time':20,'ranking':[0,2]}, {'entity':1,'time':10,'ranking':[1,3]}]
    r=fn(truth,pred,2,5)
    assert math.isclose(r['map'],.75) and r['hit']==1 and math.isclose(r['recall'],5/6)
    assert r['n']==2
    rejects(lambda:fn(truth,pred[:1],2,5))
    bad=copy.deepcopy(pred);bad[0]['ranking']=[2,2];rejects(lambda:fn(truth,bad,2,5))
    bad=copy.deepcopy(pred);bad[0]['time']=10;rejects(lambda:fn(truth,bad,2,5))
    bad=copy.deepcopy(pred);bad[0]['ranking']=[2,5];rejects(lambda:fn(truth,bad,2,5))
    bad=copy.deepcopy(truth);bad[0]['positives']=[];rejects(lambda:fn(bad,pred,2,5))

def check_negatives(fn):
    r=fn([10,10],[1,2],[{1},{2,3}],4,[0,0,11,0])
    assert r['comparisons']==4 and r['positive_collisions']==2 and r['future_comparisons']==2
    r=fn([10,10],[1,2],[{1},{2}],4)
    assert r['future_comparisons'] is None and r['availability']=='NOT_OBSERVED'
    rejects(lambda:fn([10,20],[1,2],[{1},{2}],4))
    rejects(lambda:fn([10],[4],[{1}],4))

def check_portfolio(fn):
    records=[dict(seed=i,status='COMPLETE',protocol_hash='p',counts={'val':2,'test':3},scores={'val':.2,'test':.107},temporal_violations=0) for i in range(5)]
    r=fn(records,'p',{'val':2,'test':3});assert r['paper_comparison']=='CLOSE' and r['test']['sample_sd']==0
    for boundary in (.107-.02,.107+.02):
        edge=[dict(r,scores={'val':.2,'test':boundary}) for r in records]
        assert fn(edge,'p',{'val':2,'test':3})['paper_comparison']=='CLOSE', 'Inclusive tolerance boundary'
    for outside in (.107-.02-1e-9,.107+.02+1e-9):
        edge=[dict(r,scores={'val':.2,'test':outside}) for r in records]
        assert fn(edge,'p',{'val':2,'test':3})['paper_comparison']=='OUTSIDE_TOLERANCE'
    rejects(lambda:fn(records[:4],'p',{'val':2,'test':3}))
    for field,value in [('seed',1),('status','PARTIAL_TIMING_PILOT'),('protocol_hash','q'),('temporal_violations',1),('counts',{'val':1,'test':3})]:
        bad=copy.deepcopy(records);bad[0][field]=value;rejects(lambda:fn(bad,'p',{'val':2,'test':3}))

if __name__=='__main__':
    for f,fn in [(check_ranking,ranking_metrics),(check_negatives,negative_audit),(check_portfolio,portfolio_entry)]:f(fn)
    print('PASS: keyed ranking, shared negatives, complete portfolio contracts')

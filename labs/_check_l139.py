"""Behavioral contracts, written before learner implementations."""
import numpy as np

def check_target(fn):
    # analyses: (day, p_value, p_value_modifier, outcome_type)
    assert fn(-1,[(1,.05,None,'Primary')],0)==(True,1)
    assert fn(-1,[(365,.051,None,'Primary')],0)==(True,0)
    assert fn(-1,[(0,.01,None,'Primary'),(366,.01,None,'Primary')],0)==(False,None)
    assert fn(1,[(10,.01,None,'Primary')],0)==(False,None)
    assert fn(0,[(1,.01,'>','Primary'),(2,.01,None,'Secondary')],0)==(False,None)
    assert fn(0,[(1,None,None,'Primary'),(2,-.1,None,'Primary'),(3,1.1,None,'Primary')],0)==(False,None)
    assert fn(0,[(1,.1,None,'Primary'),(2,.02,'<','Primary')],0)==(True,1)
    assert fn(0,[],0)==(False,None)
    assert fn(0,[(1,.051,'<','Primary')],0)==(True,0) # source preserves numeric value

def check_visibility(fn):
    assert fn([9,11,19,21],[0,0,1,1],[10,20]).tolist()==[True,False,True,False]
    assert fn([10,20],[0,1],[10,20]).all()
    assert fn([],[],[10]).tolist()==[]
    for times,owners,cuts in [([1],[2],[10]),([1,2],[0],[10]),([np.nan],[0],[10]),([1],[-1],[10])]:
        try:fn(times,owners,cuts)
        except ValueError:pass
        else:raise AssertionError('bad query ownership accepted')

def check_score(fn):
    keys=[(1,0),(2,0),(3,0),(4,0)];y=[0,0,1,1]
    assert fn(keys,y,keys[::-1],[2,1,1,0])==.875
    for pk,ps in [(keys[:-1],[0,1,2]),([keys[0]]*4,[0,1,2,3]),(keys[:-1]+[(9,0)],[0,1,2,3])]:
        try:fn(keys,y,pk,ps)
        except ValueError:pass
        else:raise AssertionError('invalid prediction keys accepted')
    assert fn([(1,0),(1,1)],[0,1],[(1,1),(1,0)],[.5,.5])==.5

if __name__=='__main__':
    from relkit.trial_l139 import trial_target,visibility_mask,keyed_auc
    for test,fn in [(check_target,trial_target),(check_visibility,visibility_mask),(check_score,keyed_auc)]:test(fn)
    print('PASS: target eligibility, owner-indexed cutoff, keyed AUROC')

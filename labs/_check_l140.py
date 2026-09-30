"""Behavioral contracts for the three live learner functions."""
import numpy as np

def check_selection(fn):
    h=[{'epoch':1,'val':{'roc_auc':.70},'test':.9},{'epoch':2,'val':{'roc_auc':.75},'test':.6},{'epoch':3,'val':{'roc_auc':.75},'test':1.0}]
    assert fn(h)==2, 'Use the first validation maximum; test is irrelevant'
    for bad in [[],[{'epoch':1,'val':{'roc_auc':float('nan')}}],h[:1]+h[:1]]:
        try:fn(bad)
        except ValueError:pass
        else:raise AssertionError('Reject empty, nonfinite or duplicated histories')

def check_alignment(fn):
    q=[(4,10),(4,20),(8,10)]
    assert np.array_equal(fn(q,[q[2],q[0],q[1]],[.2,.9,.4]),[.9,.4,.2]), 'Entity alone is not a query key'
    for keys,values in [(q[:2],[.1,.2]),([q[0],q[0],q[2]],[.1,.2,.3]),(q,[.1,float('nan'),.3]),(q+[q[0]],[.1,.2,.3,.4])]:
        try:fn(q,keys,values)
        except ValueError:pass
        else:raise AssertionError('Missing, duplicate, extra or invalid predictions accepted')

def check_verdict(fn):
    result=fn({0:.69,1:.71},[0,1],.70,.01,False)
    assert result['execution']=='COMPLETE' and result['score']=='CLOSE' and result['protocol']=='GAPPED'
    assert result['historical_identity']=='NOT_ESTABLISHED'
    assert abs(result['sample_sd']-np.sqrt(.0002))<1e-12
    r=fn({0:.69},[0,1],.70,.01,True)
    assert r['execution']=='INCOMPLETE' and r['score']=='NOT_EVALUATED' and r['mean'] is None
    assert fn({0:.60,1:.62},[0,1],.70,.01,True)['score']=='OUTSIDE_TOLERANCE'
    assert fn({0:.71,1:.71},[0,1],.70,.01,True)['score']=='CLOSE', 'The declared boundary is inclusive'
    assert fn({0:.71001,1:.71001},[0,1],.70,.01,True)['score']=='OUTSIDE_TOLERANCE'
    for values in [{0:.7,1:float('nan')},{0:.7,1:.7,2:.7}]:
        try:fn(values,[0,1],.70,.01,True)
        except ValueError:pass
        else:raise AssertionError('Invalid or unexpected seed accepted')

if __name__=='__main__':
    from relkit.checkpoint_l140 import select_checkpoint,align_predictions,reproduction_verdict
    for check,fn in [(check_selection,select_checkpoint),(check_alignment,align_predictions),(check_verdict,reproduction_verdict)]:check(fn)
    print('PASS: checkpoint selection, identity alignment and evidence gates')

"""Contract tests and semantic mutation checks for the reproduction lab."""
import copy,json
from pathlib import Path
import numpy as np

def check_layout(fn):
    a={'number':(12.,3.),'position':(8.,2.)}
    assert fn(['number','position'],a,[12.,8.],[3.,2.]) is True
    for cols,mean,std in [(['position','number'],[12.,8.],[3.,2.]),(['number'],[12.,8.],[3.,2.]),(['number','position'],[12.,8.],[3.,9.]),(['number','position'],[12.,float('nan')],[3.,2.])]:
        try:fn(cols,a,mean,std)
        except ValueError:pass
        else:raise AssertionError('Accepted incompatible layout')

def check_selection(fn):
    assert fn([{'epoch':1,'val_mae':4.},{'epoch':2,'val_mae':3.},{'epoch':3,'val_mae':3.}])==2
    assert fn([{'epoch':1,'val_mae':3.,'test_mae':9.},{'epoch':2,'val_mae':4.,'test_mae':1.}])==1
    for h in [[],[{'epoch':1,'val_mae':float('nan')}],[{'epoch':2,'val_mae':2.}]]:
        try:fn(h)
        except ValueError:pass
        else:raise AssertionError('Accepted invalid history')

def check_verdict(fn):
    runs=[dict(seed=i,kind='RECONSTRUCTED_TRAINING',epochs=10,complete=True,test_mae=3.798) for i in range(5)]
    r=fn(runs);assert r['execution']=='COMPLETE' and r['score']=='CLOSE' and r['historical']=='NOT_ESTABLISHED'
    assert fn(runs[::-1])==r
    assert fn(runs[:-1])['execution']=='INCOMPLETE'
    for field,value in [('kind','CHECKPOINT_COMPATIBILITY_REPLAY'),('epochs',9),('complete',False),('test_mae',float('nan')),('seed',1)]:
        bad=copy.deepcopy(runs);bad[0][field]=value
        assert fn(bad)['execution']=='INCOMPLETE',(field,value)
    assert fn([dict(x,test_mae=3.998) for x in runs])['score']=='CLOSE'
    assert fn([dict(x,test_mae=4.267) for x in runs])['score']=='OUTSIDE_TOLERANCE'

if __name__=='__main__':
    from relkit.reproduction_l143 import verify_numeric_layout,first_validation_min,evidence_verdict
    for test,fn in [(check_layout,verify_numeric_layout),(check_selection,first_validation_min),(check_verdict,evidence_verdict)]:test(fn)
    mutants=[(check_layout,lambda *x:True),(check_selection,lambda h:h[-1]['epoch']), (check_selection,lambda h:min(h,key=lambda x:x.get('test_mae',x['val_mae']))['epoch']), (check_verdict,lambda r:{'execution':'COMPLETE','score':'CLOSE','historical':'ESTABLISHED'})]
    for test,fn in mutants:
        try:test(fn)
        except (AssertionError,ValueError,IndexError):pass
        else:raise AssertionError('Surviving mutant')
    report=dict(status='PASS',contracts=3,semantic_mutants_rejected=len(mutants))
    Path(__file__).with_name('_check_l143_results.json').write_text(json.dumps(report,indent=2));print(report)

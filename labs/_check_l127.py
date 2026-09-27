"""Independent behavioral examples for the experiment ledger functions."""
import copy,json,math,statistics
from pathlib import Path
from relkit.benchmark_l127 import experiment_contract,select_checkpoint,summarize_seeds

def rejected(fn,*args,**kwargs):
    try:fn(*args,**kwargs)
    except ValueError:return
    raise AssertionError('Invalid experiment evidence accepted')
def check_contract(fn):
    c=fn();assert c['dataset']=='rel-f1' and c['task']=='driver-position'
    assert c['epochs']==10 and c['seeds']==[0,1,2,3,4]
    assert c['fanouts']==[128,64] and c['loss']=='L1' and c['learning_rate']==.005
    c['fanouts'][0]=1;assert fn()['fanouts']==[128,64], 'Shared mutable configuration'
    rejected(fn,{'epochs':1});rejected(fn,{'selection':'test_mae'});rejected(fn,{'typo':1})
    assert fn({'epochs':10})==fn()
def check_selection(fn):
    trace=[dict(epoch=i+1,train_queries=4,val_mae=v,test_mae=t) for i,(v,t) in enumerate([(3,1),(2,9),(2,.1)])]
    assert fn(trace,3,4)==2, 'First validation tie wins, test must not select'
    changed=copy.deepcopy(trace)
    for x in changed:x['test_mae']=-999
    assert fn(changed,3,4)==2
    for bad in [trace[:-1],trace[::-1],[dict(x,val_mae=float('nan')) for x in trace], [dict(x,train_queries=3) for x in trace]]:rejected(fn,bad,3,4)
def check_summary(fn):
    rows=[dict(seed=i,status='COMPLETE',run_uuid=f'run-{i}',epochs=10,val=float(i+1),test=float(2*i+1)) for i in range(5)]
    result=fn(rows);assert result['val']['mean']==3 and abs(result['val']['sample_sd']-math.sqrt(2.5))<1e-12
    assert result['test']['mean']==5 and abs(result['test']['sample_sd']-math.sqrt(10))<1e-12
    assert fn(rows[::-1])==result
    for field,value in [('seed',0),('run_uuid','run-0'),('status','RUNNING'),('epochs',1),('test',float('inf'))]:
        bad=copy.deepcopy(rows);bad[-1][field]=value;rejected(fn,bad)
    rejected(fn,rows[:-1]);rejected(fn,rows+rows[:1])
    for field in ['test','run_uuid']:
        bad=copy.deepcopy(rows);del bad[-1][field];rejected(fn,bad)
if __name__=='__main__':
    check_contract(experiment_contract);check_selection(select_checkpoint);check_summary(summarize_seeds)
    r=dict(status='PASS',checks=['immutable exact protocol','first validation minimum and test invariance','missing duplicate incomplete nonfinite seeds rejected','sample SD denominator n-1'])
    Path(__file__).with_name('_check_l127_results.json').write_text(json.dumps(r,indent=2));print(r)

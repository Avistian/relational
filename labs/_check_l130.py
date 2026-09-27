"""Behavioral contracts: deliberate reorder, same-entity queries, incomplete evidence."""
import json,math,statistics,copy
from pathlib import Path
from relkit.checkpoint_l130 import validate_queries,keyed_mae,reproduction_verdict

def rejected(fn,*args):
    try:fn(*args)
    except ValueError:return
    raise AssertionError('Invalid evidence was accepted')

def check_queries(fn):
    q=[dict(entity=7,time=10,target=2.),dict(entity=7,time=20,target=5.)]
    assert fn(q,2)==[(7,10),(7,20)],'Entity alone is not query identity'
    rejected(fn,q,3);rejected(fn,[q[0],q[0]],2);rejected(fn,[],0)
    for value in [float('nan'),float('inf')]:
        z=copy.deepcopy(q);z[0]['target']=value;rejected(fn,z,2)
    z=copy.deepcopy(q);del z[0]['time'];rejected(fn,z,2)
    z=copy.deepcopy(q);z[0]['time']=10.5;rejected(fn,z,2)

def check_mae(fn):
    q=[dict(entity=7,time=10,target=2.),dict(entity=7,time=20,target=5.)]
    p=[dict(entity=7,time=20,prediction=4.),dict(entity=7,time=10,prediction=2.)]
    assert fn(q,p)==.5,'Align by BOTH keys before scoring'
    assert fn(q,list(reversed(p)))==.5,'Row order is not identity'
    rejected(fn,q,p[:1]);rejected(fn,q,[p[0],p[0]])
    z=copy.deepcopy(p);z[0]['time']=30;rejected(fn,q,z)
    z=copy.deepcopy(p);z[0]['prediction']=float('nan');rejected(fn,q,z)
    rejected(fn,[q[0],q[0]],p)

def check_verdict(fn):
    r=[dict(seed=s,run_uuid=f'run-{s}',status='COMPLETE',epochs=10,val=3.193,test=4.022) for s in range(5)]
    out=fn(r);assert out['val']['verdict']==out['test']['verdict']=='CLOSE'
    assert out['test']['sample_sd']==0 and out['test']['n']==5
    z=copy.deepcopy(r);z[0]['test']=5.522
    assert fn(z)['test']['verdict']=='OUTSIDE_TOLERANCE','Do not manufacture a match'
    assert math.isclose(fn(z)['test']['sample_sd'],statistics.stdev(x['test'] for x in z))
    rejected(fn,r[:-1]);rejected(fn,[r[0]]*5)
    for field,value in [('epochs',9),('status','RUNNING'),('run_uuid','run-1'),('test',float('nan')),('val',-1)]:
        z=copy.deepcopy(r);z[0][field]=value;rejected(fn,z)
    rejected(fn,r,(0,1,2,3,4),-.1)

if __name__=='__main__':
    for fn,check in [(validate_queries,check_queries),(keyed_mae,check_mae),(reproduction_verdict,check_verdict)]:check(fn)
    out=dict(status='PASS',tasks=3,checks='Repeated entity at different times; shuffled predictions; missing/duplicate/unknown keys; nonfinite values; incomplete/reused seeds; outside tolerance')
    Path(__file__).with_name('_check_l130_results.json').write_text(json.dumps(out,indent=2));print(out)

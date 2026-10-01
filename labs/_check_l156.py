"""Behavioral checks include leaks that a max-batch-cutoff or binary checklist misses."""
import copy,json
from pathlib import Path
from relkit.temporal_audit_l156 import audit_observations,audit_label_windows,audit_verdict

def rejects(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid evidence accepted')

def observations():
    return [dict(owner=0,cutoff=10,event=10,available=10,rule='inclusive',kind='event'),
            dict(owner=1,cutoff=20,event=21,available=18,rule='inclusive',kind='schedule'),
            dict(owner=0,cutoff=10,event=9,available=None,rule='strict',kind='event')]

def check_observations(fn):
    r=fn(observations());assert r['status']=='NOT_ESTABLISHED' and r['counts']=={'PASS':2,'FAIL':0,'NOT_ESTABLISHED':1}
    cases=[('event',11),('available',11),('rule','strict')]
    for field,value in cases:
        x=observations();x[0][field]=value
        result=fn(x);assert result['status']=='FAIL' and result['rows'][0]['status']=='FAIL'
    x=observations();x[0]['available']=None;x[0]['event']=11
    assert fn(x)['status']=='FAIL','Known future cannot be excused by missing arrival time'
    x=observations();x[1]['available']=21;assert fn(x)['rows'][1]['status']=='FAIL'
    for field,value in [('event',float('nan')),('cutoff',float('inf')),('rule','maybe'),('kind','anything')]:
        x=observations();x[0][field]=value;rejects(lambda:fn(x))
    x=observations();x[2]['cutoff']=12;rejects(lambda:fn(x))
    rejects(lambda:fn([]))

def check_labels(fn):
    q=[dict(entity=1,time=10,target=4.),dict(entity=1,time=20,target=8.)]
    e=[dict(entity=1,time=10,value=99.),dict(entity=1,time=11,value=2.),dict(entity=1,time=15,value=6.),dict(entity=1,time=21,value=8.),dict(entity=1,time=26,value=99.)]
    r=fn(q,e,5);assert r['status']=='PASS' and r['queries']==2 and r['label_events']==3
    x=copy.deepcopy(q);x[0]['target']=99.;assert fn(x,e,5)['status']=='FAIL'
    x=copy.deepcopy(q);x[0]['time']=15;assert fn(x,e,5)['status']=='FAIL'
    rejects(lambda:fn(q+q[:1],e,5));rejects(lambda:fn(q,e,0));rejects(lambda:fn([],e,5))
    x=copy.deepcopy(q);x[0]['target']=float('nan');rejects(lambda:fn(x,e,5))

def check_verdict(fn):
    required=['cutoff','arrival','labels']
    c=[dict(id='cutoff',status='PASS',evidence='sha256:a'),dict(id='arrival',status='NOT_ESTABLISHED',evidence='missing history'),dict(id='labels',status='PASS',evidence='sha256:b')]
    assert fn(c,required)['status']=='NOT_ESTABLISHED'
    assert fn(c[:1],required)['status']=='NOT_CHECKED'
    x=copy.deepcopy(c);x[0]['status']='FAIL';assert fn(x,required)['status']=='FAIL'
    x=copy.deepcopy(c);x[1]['status']='PASS';assert fn(x,required)['status']=='PASS'
    x=copy.deepcopy(c);x[0]['evidence']='';rejects(lambda:fn(x,required))
    rejects(lambda:fn(c+c[:1],required));rejects(lambda:fn(c,[]))
    x=copy.deepcopy(c);x[0]['status']='COMPLETE';rejects(lambda:fn(x,required))
    x=copy.deepcopy(c);x[0]['status']='FAIL';assert fn(x[:1],required)['status']=='FAIL'

if __name__=='__main__':
    check_observations(audit_observations);check_labels(audit_label_windows);check_verdict(audit_verdict)
    out=dict(status='PASS',contracts=['owner cutoff and availability','future label window','coverage and evidence sign-off'],fixtures='SYNTHETIC')
    Path(__file__).with_name('_check_l156_results.json').write_text(json.dumps(out,indent=2));print(out)

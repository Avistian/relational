"""Behavioral contracts; fixtures are synthetic, never human-study measurements."""
import copy,json
from pathlib import Path
from relkit.effort_l155 import paired_losses,summarize_effort,effort_ratio

def rejects(call):
    try:call()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid evidence was accepted')

def check_pairs(fn):
    q=[dict(entity=1,time=10,target=4.),dict(entity=1,time=20,target=8.)]
    f=[dict(entity=1,time=20,prediction=6.),dict(entity=1,time=10,prediction=3.)]
    g=[dict(entity=1,time=10,prediction=6.),dict(entity=1,time=20,prediction=8.)]
    r=fn(q,f,g);assert r['fe_mae']==1.5 and r['rdl_mae']==1. and r['benefit_mae']==.5
    assert [x['benefit'] for x in r['rows']]==[-1.,2.]
    rejects(lambda:fn(q,f[:1],g));rejects(lambda:fn(q,f+[f[0]],g));rejects(lambda:fn(q+[q[0]],f,g))
    for field,value in [('prediction',float('nan')),('time',21)]:
        bad=copy.deepcopy(g);bad[0][field]=value;rejects(lambda:fn(q,f,bad))
    bad=copy.deepcopy(q);bad[0]['target']=float('inf');rejects(lambda:fn(bad,f,g))
    rejects(lambda:fn([],[],[]))

def fixture():
    common=dict(task='demo',participant='learner',assistance='none',kind='human',scope='marginal',phase='feature_implementation')
    return [dict(common,id='a',method='FE',start='2026-10-01T08:00:00+00:00',end='2026-10-01T10:00:00+00:00'),
            dict(common,id='b',method='RDL',start='2026-10-01T11:00:00+00:00',end='2026-10-01T11:30:00+00:00'),
            dict(common,id='c',method='RDL',scope='shared',phase='setup',start='2026-10-01T12:00:00+00:00',end='2026-10-01T13:30:00+00:00'),
            dict(common,id='d',method='FE',kind='machine',phase='training',start='2026-10-01T09:00:00+00:00',end='2026-10-01T12:00:00+00:00')]

def check_effort(fn):
    coverage={'FE':'complete','RDL':'complete'}
    invoke=lambda ss,c=coverage:fn(ss,'demo','learner','none',c)
    r=invoke(fixture());assert r['human_marginal_hours']=={'FE':2.,'RDL':.5}
    assert r['human_shared_hours']=={'FE':0.,'RDL':1.5} and r['machine_hours']['FE']==3
    absent=invoke([],{'FE':'not_observed','RDL':'not_observed'});assert absent['human_marginal_hours']=={'FE':None,'RDL':None}
    rejects(lambda:invoke([],coverage))
    for field,value in [('end',None),('end','2026-10-01T07:00:00+00:00'),('start','2026-10-01T08:00:00'),('assistance','agent'),('scope','unknown'),('kind','agent'),('phase',''),('task','other')]:
        bad=fixture();bad[0][field]=value;rejects(lambda:invoke(bad))
    bad=fixture();bad[1]['start']='2026-10-01T09:00:00+00:00';rejects(lambda:invoke(bad))
    rejects(lambda:invoke(fixture()+[fixture()[0]]))
    partial=invoke(fixture(),{'FE':'partial','RDL':'complete'});assert partial['coverage']['FE']=='partial'

def check_ratio(fn):
    base=dict(task='demo',participant='learner',assistance='none',coverage={'FE':'complete','RDL':'complete'},human_marginal_hours={'FE':2.,'RDL':.5})
    assert fn(base)==dict(status='OBSERVED_DESCRIPTIVE',ratio=4.,reduction_percent=75.)
    for hours in [{'FE':None,'RDL':None},{'FE':2.,'RDL':0.}]:
        x=copy.deepcopy(base);x['human_marginal_hours']=hours;assert fn(x)['ratio'] is None
    x=copy.deepcopy(base);x['coverage']['FE']='partial';assert fn(x)['status']=='INCOMPLETE'
    x=copy.deepcopy(base);x['human_marginal_hours']['FE']=-1.;rejects(lambda:fn(x))
    x=copy.deepcopy(base);x['human_marginal_hours']['FE']=float('nan');rejects(lambda:fn(x))

if __name__=='__main__':
    check_pairs(paired_losses);check_effort(summarize_effort);check_ratio(effort_ratio)
    r=dict(status='PASS',contracts=['keyed paired loss','observed effort with nonoverlap','ratio completeness'],fixture_origin='SYNTHETIC')
    Path(__file__).with_name('_check_l155_results.json').write_text(json.dumps(r,indent=2));print(r)

"""Behavioral contracts for the three learner functions. No training required."""
from copy import deepcopy

def example_record():
    return dict(pretraining_databases=['A','B'],evaluation_database='C',
                backbone_updates=0,head_updates=0,target_labels=8,
                declared_mode='IN_CONTEXT',selection_split='validation',evaluation_split='test',
                test_labels_used=False,temporal_audit='PASS',baseline_matched=True,
                completed=False,checkpoint_shared=True)

def check161(route,boundary,verdict):
    assert route(0,0,0)=='ZERO_SHOT'
    assert route(0,0,8)=='IN_CONTEXT'
    assert route(0,20,8)=='FROZEN_ENCODER_HEAD'
    assert route(20,20,8)=='FINE_TUNING'
    assert route(1,0,0)=='FINE_TUNING', 'Unsupervised adaptation can update weights without target labels'
    for bad in [-1,True,0.5,'8',None]:
        try: route(0,0,bad)
        except ValueError: pass
        else: raise AssertionError('Invalid count accepted')
    assert boundary(['A','B'],'C')=='HELD_OUT'
    assert boundary(['A','B'],'A')=='SEEN'
    assert boundary(None,'C')=='UNKNOWN'
    assert boundary([],'C')=='UNKNOWN', 'Empty corpus does not document broad pretraining'
    for corpus,target in [(['A','A'],'C'),([' A'],'C'),('AB','C'),(['A'],''),([1],'C')]:
        try: boundary(corpus,target)
        except ValueError: pass
        else: raise AssertionError('Ambiguous database identity accepted')
    base=example_record()
    assert verdict(base)['status']=='READY_TO_RUN', 'A plan is not a measured experiment'
    done=deepcopy(base);done['completed']=True
    assert verdict(done)['status']=='READY_FOR_REVIEW'
    assert verdict(done)['universal_claim']=='NOT_ESTABLISHED'
    assert verdict(done)['learner']=='PENDING_WRITTEN_DEFENSE'
    changes=[('pretraining_databases',None,'PRETRAINING_UNKNOWN'),
             ('pretraining_databases',['A','C'],'PRETRAINING_OVERLAP'),
             ('selection_split','test','TEST_SELECTION'),
             ('evaluation_split','validation','NO_HELD_OUT_TEST'),
             ('test_labels_used',True,'TEST_LABEL_ACCESS'),
             ('temporal_audit','NOT_ESTABLISHED','TEMPORAL_UNVERIFIED'),
             ('baseline_matched',False,'BASELINE_UNMATCHED'),
             ('checkpoint_shared',False,'NO_SHARED_CHECKPOINT'),
             ('declared_mode','ZERO_SHOT','MODE_MISMATCH')]
    for field,value,issue in changes:
        r=deepcopy(done);r[field]=value
        answer=verdict(r)
        assert answer['status']=='REVISE' and issue in answer['issues'], (field,answer)
    assert base==example_record(), 'Auditor must not mutate input'
    for field,value in [('completed','yes'),('temporal_audit','maybe'),('target_labels',False),('declared_mode','magic')]:
        r=deepcopy(base);r[field]=value
        try: verdict(r)
        except ValueError: pass
        else: raise AssertionError('Malformed record accepted: '+field)
    observed=[]
    def routed(*args): observed.append('route');return route(*args)
    def bounded(*args): observed.append('boundary');return boundary(*args)
    verdict(base,route=routed,boundary=bounded)
    assert observed==['route','boundary'], 'The real adapter must call the learner functions'
    return 'PASS'

if __name__=='__main__':
    from relkit.scope_l161 import adaptation_route,database_boundary,scope_verdict
    print(check161(adaptation_route,database_boundary,scope_verdict))

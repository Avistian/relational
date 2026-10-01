"""Adversarial exam contracts; synthetic fixtures are never benchmark evidence."""
import copy
from relkit.exam_l160 import aligned_losses, observed_effort, exit_gates

def rejects160(fn,*args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError('Invalid evidence accepted')

def check160(align=aligned_losses,effort=observed_effort,gates=exit_gates):
    keys=[(7,10),(7,20),(8,10)]
    r=align(keys,[1,3,5],keys[::-1],[7,4,1],keys,[2,5,5])
    assert r['fe_loss']==[0.,1.,2.] and r['rdl_loss']==[1.,2.,0.]
    assert r['benefit']==[-1.,-1.,2.] and r['mean_benefit']==0
    rejects160(align,keys,[1,3,5],[keys[0]]*3,[1,4,7],keys,[2,5,5])
    rejects160(align,keys,[1,3,5],keys[:-1],[1,4],keys,[2,5,5])
    rejects160(align,keys,[1,3,5],keys,[1,float('nan'),7],keys,[2,5,5])
    rejects160(align,[],[],[],[],[],[])
    assert effort([])=={'status':'NOT_OBSERVED','ratio_fe_over_rdl':None,'minutes':{}}
    logs=[dict(id='f',arm='FE',task='x',scope='build+debug+validate',minutes=90,kind='human_active',prospective=True),dict(id='r',arm='RDL',task='x',scope='build+debug+validate',minutes=30,kind='human_active',prospective=True)]
    assert effort(logs)['ratio_fe_over_rdl']==3
    assert effort(logs[:1])['status']=='INCOMPLETE'
    rejects160(effort,logs+[logs[0]])
    for change in [dict(minutes=-1),dict(minutes=float('inf')),dict(kind='gpu_runtime'),dict(prospective=False),dict(scope='training only'),dict(task='other')]:
        rejects160(effort,[logs[0],dict(logs[1],**change)])
    zero=copy.deepcopy(logs);zero[1]['minutes']=0
    assert effort(zero)['status']=='UNDEFINED_ZERO_DENOMINATOR'
    entries=[dict(task=t,status='COMPLETE',split='test',matched_fe='COMPLETE',effort='OBSERVED',temporal='PASS') for t in ['db/a','db/b','other/c']]
    failures=[dict(evidence='saved/negative-result.json',limitation='FE point estimate wins; interval crosses zero')]
    review=dict(reviewer='teacher',scores=[2,2,2,1,1])
    assert gates(entries,failures,review)['exit']=='PASS'
    assert gates(entries,failures)['exit']=='PENDING_WRITTEN_DEFENSE'
    assert gates(entries,[],review)['exit']=='INCOMPLETE'
    bad=copy.deepcopy(entries);bad[2]['status']='INCOMPLETE'
    assert gates(bad,failures,review)['counts']['completed_tasks']==2
    assert gates(bad,failures,review)['exit']=='INCOMPLETE'
    for field,value in [('matched_fe','NOT_RUN'),('effort','NOT_OBSERVED'),('temporal','NOT_ESTABLISHED'),('split','val')]:
        bad=copy.deepcopy(entries);bad[2][field]=value
        assert gates(bad,failures,review)['exit']=='INCOMPLETE',field
    rejects160(gates,entries+[entries[0]],failures,review)
    rejects160(gates,entries,failures,dict(reviewer='teacher',scores=[2,2,2,2,3]))
    assert gates(entries,failures,dict(reviewer='teacher',scores=[2,2,2,2,0]))['exit']=='REVISION_REQUIRED'
    # A fourth complete task cannot conceal failure of a declared portfolio member.
    bad=copy.deepcopy(entries);bad.append(dict(entries[0],task='extra/d',temporal='FAIL'))
    assert gates(bad,failures,review)['exit']=='INCOMPLETE'
    return dict(status='PASS',mechanisms=3)

if __name__=='__main__': print(check160())

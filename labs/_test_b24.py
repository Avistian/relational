"""Behavioral contracts shared by the visible learner lab and independent audit."""
import copy, itertools
from relkit.defense_b24 import paired_effect, defense_gate, falsification_contract
AXES=['protocol','baselines','reproducibility','interpretation','falsifiability']
def fixtures():
    # Deliberately shuffled; positional pairing would give the wrong sign count.
    rows=[dict(arm=a,seed=s,auc=v) for a,vs in [('A',[.7,.6,.9]),('B',[.8,.5,.7])] for s,v in enumerate(vs)]
    tests=[dict(kind='simpler_baseline',task='new-db/new-task',challenger='relational model',reference='tuned trees + time-safe FE',metric='AUROC',orientation='higher',threshold=0.,information_matched=True,budget_matched=True,action='Drop the superiority claim if mean challenger-minus-reference <= 0.'),dict(kind='untouched_task',task='new-db/new-task',challenger='relational model',reference='locked strongest baseline',metric='AUROC',orientation='higher',threshold=0.,information_matched=True,budget_matched=True,action='Narrow the transfer claim if the held-out task effect <= 0.')]
    return rows,tests

def rejects(fn,*args):
    try:fn(*args)
    except (ValueError,TypeError):return
    raise AssertionError('Invalid input accepted')

def checks(pair=paired_effect,gate=defense_gate,falsify=falsification_contract):
    rows,tests=fixtures();result=pair(list(reversed(rows)), 'B','A')
    assert result['seeds']==[0,1,2] and result['positive']==2
    assert abs(result['mean']-1/15)<1e-12
    assert abs(result['per_seed'][0]+.1)<1e-12
    for bad in [rows[:-1],rows+[rows[0]],rows+[dict(arm='A',seed=4,auc=float('nan'))]]:rejects(pair,bad,'B','A')
    rejects(pair,rows,'A','A')
    for vals in itertools.product(range(3),repeat=5):
        scores=dict(zip(AXES,vals));quality=sum(vals)>=8 and min(vals)>0
        for repro,leak,assessed in itertools.product([False,True],repeat=3):
            r=gate(scores,repro,leak,assessed)
            assert r['eligible']==(quality and repro and not leak)
            assert r['learner']==('PASS' if r['eligible'] else 'REVISION_REQUIRED') if assessed else r['learner']=='PENDING_WRITTEN_DEFENSE'
    good=dict.fromkeys(AXES,2)
    for bad in [dict(protocol=2),dict(good,protocol=True),dict(good,protocol=3),dict(good,extra=2)]:rejects(gate,bad,True,False,False)
    rejects(gate,good,'yes',False,False)
    assert falsify(tests,['rel-f1/driver-dnf'])=={'ready':True,'tests':2,'execution':'NOT_RUN'}
    for field,value in [('task','rel-f1/driver-dnf'),('threshold',float('nan')),('information_matched',False),('budget_matched',False),('action',''),('reference',''),('orientation','unknown')]:
        bad=copy.deepcopy(tests);bad[1][field]=value;rejects(falsify,bad,['rel-f1/driver-dnf'])
    rejects(falsify,tests[:1],[])
    bad=copy.deepcopy(tests);bad[1]['kind']='simpler_baseline';rejects(falsify,bad,[])
    return 'PASS'
if __name__=='__main__':print(checks())

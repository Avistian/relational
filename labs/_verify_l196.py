"""Independent sorted-vocabulary oracle and hostile packet/learner checks."""
import copy,hashlib,json,shutil,tempfile,os,subprocess
from pathlib import Path
from _report_l196 import make_report
from _test_l196 import check
from relkit.community_l196 import summarize_cases,route_question,feedback_state
P=Path(__file__).resolve().parent;E=P/'evidence/l196';d=json.loads((E/'diagnostic.json').read_text())
assert d['status']=='COMPLETE_DIAGNOSTIC' and len(d['observations'])==4
for case in d['observations']:
    u=case['unseen'];vocab=sorted(['b','c','d',u])
    assert case['before']==[0,1,2]
    assert case['after']==[vocab.index(x) for x in ['b','c','d']]
    assert case['numeric_before']==case['numeric_after']==[1,2,3]
    assert case['query_alone']==[0]
    assert case['query_with_other']==[vocab.index(u),vocab.index('b')]
    assert case['cached_support']==[0,1,2]*4
assert make_report(E)==json.loads((E/'report.json').read_text())
checks=check();rejected=0
for bad in [lambda _:dict(model_effect='LOWER_AUROC'),lambda _:dict(cases=1),lambda _:dict(cases=4,changed_cases=[],batch_sensitive_cases=[],numeric_controls_unchanged=True,model_effect='NOT_ESTABLISHED')]:
    try:check(summarize=bad)
    except AssertionError:rejected+=1
    else:raise AssertionError('Wrong evidence function accepted')
try:check(feedback=lambda *_:'FEEDBACK_CHECKED')
except AssertionError:rejected+=1
else:raise AssertionError('Unposted feedback accepted')
with tempfile.TemporaryDirectory() as tmp:
    q=Path(tmp);shutil.copytree(E/'packet',q/'packet');shutil.copy(E/'diagnostic.json',q/'diagnostic.json')
    file=q/'packet/source/rdblearn/rdblearn/preprocessing.py';file.write_text(file.read_text()+'\n# changed\n')
    try:make_report(q)
    except ValueError:rejected+=1
    else:raise AssertionError('Altered source admitted')
env=dict(os.environ,PYTHONPATH=str(E/'packet/source/rdblearn')+os.pathsep+str(E/'packet/source'))
minimal=subprocess.run(['/tmp/l192-repro-env/bin/python',str(E/'minimal-question.py')],env=env,capture_output=True,text=True,check=True)
assert minimal.stdout.splitlines()==['a [0, 1, 2] [1, 2, 3]','z [0, 1, 2] [0, 1, 2]','0 [0, 1, 2] [1, 2, 3]','e [0, 1, 2] [0, 1, 2]']
result=dict(minimal_question_snippet='EXECUTED_EXACT_OUTPUT',status='PASS' ,full_original_cases=4,independent_sorted_vocabulary_oracle='PASS',numeric_controls=4,query_batch_checks=4,hostile_policies_and_source_rejected=rejected,learner_contracts=checks)
(P/'_verify_l196_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

"""Independent truth-table checks, adversarial behavior and provenance receipts."""
import hashlib,json,platform,signal
from pathlib import Path
from itertools import product
from _check_l161 import check161
from _run_l161 import audit161
from relkit.scope_l161 import adaptation_route,database_boundary,scope_verdict
signal.alarm(600)
P=Path(__file__).resolve().parent;E=P/'evidence/l161'
raw=(E/'fixtures.json').read_bytes();manifest=json.loads((E/'input-manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest()==manifest['fixtures_sha256']
packet=json.loads(raw)
assert check161(adaptation_route,database_boundary,scope_verdict)=='PASS'
report=audit161(packet,adaptation_route,database_boundary,scope_verdict)
assert all(r['status']==case['expected'] for r,case in zip(report['rows'],packet['cases']))
# Independent exhaustive oracle: index a predeclared table, not the function's if-chain.
table=['ZERO_SHOT','IN_CONTEXT','FROZEN_ENCODER_HEAD','FROZEN_ENCODER_HEAD','FINE_TUNING','FINE_TUNING','FINE_TUNING','FINE_TUNING']
for b,h,l in product([0,1,9],repeat=3):
    index=4*bool(b)+2*bool(h)+bool(l)
    assert adaptation_route(b,h,l)==table[index]
# Every subset of eight simultaneous disqualifiers must preserve every issue; no early exit.
from _check_l161 import example_record
changes=[('pretraining_databases',None),('selection_split','test'),('evaluation_split','validation'),
         ('test_labels_used',True),('temporal_audit','FAIL'),('baseline_matched',False),
         ('checkpoint_shared',False),('declared_mode','ZERO_SHOT')]
for mask in range(256):
    record=example_record();record['completed']=True
    for i,(key,value) in enumerate(changes):
        if mask&(1<<i):record[key]=value
    result=scope_verdict(record)
    assert len(result['issues'])==mask.bit_count()
    assert result['status']==('REVISE' if mask else 'READY_FOR_REVIEW')
# Deliberately incorrect learner implementations must fail the behavioral suite.
mutants=[(lambda b,h,l:'ZERO_SHOT',database_boundary,scope_verdict),
         (adaptation_route,lambda p,e:'HELD_OUT',scope_verdict),
         (adaptation_route,database_boundary,lambda r,**kw:dict(status='READY_FOR_REVIEW'))]
for funcs in mutants:
    try:check161(*funcs)
    except (AssertionError,KeyError,ValueError):pass
    else:raise AssertionError('Incorrect learner implementation escaped')
assert json.loads((E/'report.json').read_text())==report
receipt=dict(status='PASS',fixture_cases=len(packet['cases']),adaptation_oracle_cases=27,
             combined_issue_cases=256,incorrect_implementations_rejected=3,
             python=platform.python_version(),cloud_spend_usd=0,
             transfer_performance='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',
             sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['relkit/scope_l161.py','_check_l161.py','_run_l161.py','evidence/l161/fixtures.json']})
(P/'_verify_l161_results.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)

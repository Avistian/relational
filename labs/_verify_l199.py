"""Independent inherited oracles plus rational decision and corruption checks."""
import copy,hashlib,itertools,json,os,shutil,subprocess,sys,tempfile
from fractions import Fraction
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l199';Q=E/'packet'
sys.dont_write_bytecode=True
import relkit
relkit.__path__.insert(0,str(Q/'relkit'));sys.path.insert(0,str(Q))
from _replay_l190 import replay190
from relkit.checkpoint_l190 import keyed_auc,claim_gate,rank_cases
from relkit.direction_l199 import priority,launch_gate,memo_readiness
from _audit_l199 import audit199
from _test_l199 import checks
args=(replay190,keyed_auc,claim_gate,rank_cases,priority,launch_gate,memo_readiness)
r=json.loads((E/'report.json').read_text());assert audit199(E,*args)==r
checks(priority,launch_gate,memo_readiness)
with tempfile.TemporaryDirectory(prefix='l199-independent-') as td:
    dest=Path(td)/'labs';shutil.copytree(Q,dest);(dest/'relkit/__init__.py').write_text('')
    subprocess.run([sys.executable,str(dest/'_verify_l190.py')],check=True,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    inherited=json.loads((dest/'_verify_l190_results.json').read_text())
cases=json.loads((Q/'evidence/l190/packet/cases.json').read_text())
def oracle(cases,w):
    values={c['id']:Fraction(c['impact'],sum(w))*sum(Fraction(a)*b for a,b in zip(c['feasibility'],w)) for c in cases}
    return dict(scores={k:float(v) for k,v in values.items()},leaders=sorted(k for k,v in values.items() if v==max(values.values())))
for row in r['weight_sensitivity']:assert {k:row[k] for k in ['scores','leaders']}==oracle(cases,row['weights'])
for row in r['rating_sensitivity']:
    cc=copy.deepcopy(cases);c=next(c for c in cc if c['id']==row['candidate'])
    if row['field']=='impact':c['impact']=row['after']
    else:c['feasibility'][['data','implementation','compute'].index(row['field'])]=row['after']
    assert {k:row[k] for k in ['scores','leaders']}==oracle(cc,[1,1,1])
assert r['launch']['state']=='DO_NOT_LAUNCH' and r['memo']['state']=='DRAFT'
wrong=[(lambda *a:dict(scores={},leaders=['availability']),launch_gate,memo_readiness),(priority,lambda g:dict(state='READY_FOR_REVIEW',blockers=[],authorization='NOT_GRANTED'),memo_readiness),(priority,launch_gate,lambda s:dict(state='READY_FOR_REVIEW',missing=[],mastery='PASS'))]
for functions in wrong:
    try:checks(*functions)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Wrong learner implementation passed')
for mutation in ['bytes','missing','extra','promoted_receipt']:
    with tempfile.TemporaryDirectory(prefix='l199-corrupt-') as td:
        e=Path(td)/'evidence';shutil.copytree(E,e,ignore=shutil.ignore_patterns('reproducer.zip'))
        q=e/'packet';path=q/'receipts/l194.json'
        if mutation=='bytes':path.write_text('{}')
        elif mutation=='missing':path.unlink()
        elif mutation=='extra':(q/'extra.txt').write_text('unexpected')
        else:
            d=json.loads(path.read_text());d['task_results'][0]['fresh_mean']=0.9;path.write_text(json.dumps(d))
            m=json.loads((e/'input-manifest.json').read_text());h=hashlib.sha256(path.read_bytes()).hexdigest();m['files']['receipts/l194.json']=h;m['origins']['receipts/l194.json']['sha256']=h;(e/'input-manifest.json').write_text(json.dumps(m))
        try:audit199(e,*args)
        except ValueError:pass
        else:raise AssertionError('Corruption accepted: '+mutation)
result=dict(status='PASS',report_parity='EXACT',inherited_independent_verifier=inherited,weight_scenarios=27,rating_scenarios=len(r['rating_sensitivity']),mandatory_gate_states=81,wrong_learner_functions_rejected=3,new_corruptions_rejected=4,later_receipts='AUTHENTICATED_ONLY')
(P/'_verify_l199_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

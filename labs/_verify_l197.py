"""Independent predecessor oracles run on frozen copies; reject damaged evidence."""
import copy,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from _audit_l197 import audit197
from relkit.landscape_l197 import coverage_report,admit_claim,essay_readiness
from _test_l197 import check
P=Path(__file__).resolve().parent;E=P/'evidence/l197';Q=E/'packet'
# Run frozen independently implemented oracles in a disposable checkout, never overwrite predecessors.
with tempfile.TemporaryDirectory(prefix='l197-independent-') as tmp:
    root=Path(tmp);shutil.copytree(Q,root,dirs_exist_ok=True)
    results={}
    for lesson in ['l191','l195']:
        proc=subprocess.run([sys.executable,str(root/('_verify_'+lesson+'.py'))],cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,check=True)
        results[lesson]=json.loads((root/('_verify_'+lesson+'_results.json')).read_text())
        assert results[lesson]['status']=='PASS'
# Import the exact pinned implementation after independent oracle processes.
sys.dont_write_bytecode=True
import relkit
relkit.__path__.insert(0,str(Q/'relkit'));sys.path.insert(0,str(Q))
from _replay_l191 import replay191
from _replay_l195 import replay195
from relkit.stress_l195 import paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval
args=[replay191,replay195,paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval,coverage_report,admit_claim]
report=json.loads((E/'report.json').read_text());assert audit197(E,*args)==report
rejected=0
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);shutil.copytree(Q,root/'packet');shutil.copy(E/'input-manifest.json',root/'input-manifest.json')
    path=root/'packet/evidence/l191/packet/tables.json';original=path.read_bytes()
    for mode in ['corrupt','missing','extra']:
        path.write_bytes(original)
        if mode=='corrupt':path.write_bytes(original+b' ')
        elif mode=='missing':path.unlink()
        else:(root/'packet/extra.json').write_text('{}')
        try:audit197(root,*args)
        except ValueError:rejected+=1
        else:raise AssertionError('Corruption accepted: '+mode)
checks=check(coverage_report,admit_claim,essay_readiness)
# Each student function is necessary: these plausible wrong implementations must fail.
mutations=[(lambda expected,rows:dict(declared=len(expected),measured=len(rows),missing=[]),admit_claim,essay_readiness),
           (coverage_report,lambda *args:'ADMISSIBLE_SCOPED',essay_readiness),
           (coverage_report,admit_claim,lambda sections:dict(state='READY_FOR_REVIEW',missing=[],mastery='MASTERED'))]
for funcs in mutations:
    try:check(*funcs)
    except AssertionError:pass
    else:raise AssertionError('Wrong learner implementation passed')
result=dict(status='PASS',independent=results,contracts=checks,rejected_packet_corruptions=rejected,learner_mutations=3,full_report_parity='EXACT',fresh_models='NOT_RUN')
(P/'_verify_l197_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

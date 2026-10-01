"""Enforce the approved aggregate 600-second local numerical execution ceiling."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ledger=P/'evidence/l170/local-budget.json'
if __name__=='__main__':
    state=json.loads(ledger.read_text()) if ledger.exists() else dict(cap_seconds=600,cloud_usd=0,attempts=[dict(command='initial RED contract check',seconds=1.133,status='EXPECTED_FAIL')])
    if state.get('active'):raise SystemExit('Prior attempt interrupted: reconcile its reservation before continuing')
    remaining=state['cap_seconds']-sum(a['seconds'] for a in state['attempts'])
    if remaining<=0:raise SystemExit('INCOMPLETE_LOCAL_BUDGET_GATE')
    command=sys.argv[1:]
    if not command:raise SystemExit('Provide a command')
    state['active']=dict(command=command,reserved_seconds=remaining);ledger.write_text(json.dumps(state,indent=2)+'\n')
    started=time.monotonic();code=124
    try:
        result=subprocess.run(command,timeout=remaining,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'));code=result.returncode
    finally:
        state.pop('active');state['attempts'].append(dict(command=command,seconds=time.monotonic()-started,status='PASS' if code==0 else 'FAIL_OR_TIMEOUT'))
        ledger.write_text(json.dumps(state,indent=2)+'\n')
    raise SystemExit(code)

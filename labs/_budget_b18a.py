"""Bound local numerical runs, including failures and portable notebook checks."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
    ledger=P/'evidence/b18a/local-budget.json'
    state=json.loads(ledger.read_text()) if ledger.exists() else dict(cap_seconds=3600,attempts=[dict(command=['initial RED/GREEN unit checks'],seconds=10,status='ACCOUNTED_CONSERVATIVE')])
    if state.get('active'):raise SystemExit('Reconcile interrupted reservation')
    left=state['cap_seconds']-sum(a['seconds'] for a in state['attempts'])
    if left<=0:raise SystemExit('INCOMPLETE_LOCAL_BUDGET_GATE')
    command=sys.argv[1:];state['active']=dict(command=command,reserved_seconds=left);ledger.write_text(json.dumps(state,indent=2))
    start=time.monotonic();code=124
    try:
        p=subprocess.Popen(command,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        try:code=p.wait(timeout=left)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    finally:
        state.pop('active');state['attempts'].append(dict(command=command,seconds=time.monotonic()-start,status='PASS' if code==0 else 'FAIL'))
        ledger.write_text(json.dumps(state,indent=2)+'\n')
    raise SystemExit(code)

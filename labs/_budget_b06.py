"""Account all numerical attempts, including failure; kill descendants at cutoff."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
 ledger=P/'evidence/b06/local-budget.json';ledger.parent.mkdir(parents=True,exist_ok=True)
 state=json.loads(ledger.read_text()) if ledger.exists() else dict(cap_seconds=3600,cloud_usd=0,attempts=[])
 if state.get('active'):raise SystemExit('Interrupted reservation requires reconciliation')
 remaining=state['cap_seconds']-sum(x['seconds'] for x in state['attempts'])
 if remaining<=0:raise SystemExit('INCOMPLETE_LOCAL_BUDGET_GATE')
 command=sys.argv[1:]
 if not command:raise SystemExit('Command required')
 state['active']=dict(command=command,reserved_seconds=remaining);ledger.write_text(json.dumps(state,indent=2)+'\n')
 start=time.monotonic();code=124
 try:
  proc=subprocess.Popen(command,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  try:code=proc.wait(timeout=remaining)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 finally:
  state.pop('active');state['attempts'].append(dict(command=command,seconds=time.monotonic()-start,status='PASS' if code==0 else 'FAIL_OR_TIMEOUT'))
  ledger.write_text(json.dumps(state,indent=2)+'\n')
 raise SystemExit(code)

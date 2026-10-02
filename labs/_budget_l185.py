"""Account every author numerical command, including failures, against 600 seconds."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent/'evidence/l185/budget.json'
if __name__=='__main__':
 with P.with_suffix('.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  state=json.loads(P.read_text()) if P.exists() else {'limit_seconds':600,'cloud_usd':0,'jobs':[]}
  remaining=600-sum(j['seconds'] for j in state['jobs'])
  if remaining<=0:raise SystemExit('INCOMPLETE: numerical budget exhausted')
  start=time.monotonic();rc=1
  try:rc=subprocess.run(sys.argv[1:],timeout=remaining).returncode
  except subprocess.TimeoutExpired:rc=124
  finally:
   state['jobs'].append({'command':sys.argv[1:],'seconds':time.monotonic()-start,'returncode':rc})
   P.write_text(json.dumps(state,indent=2)+'\n')
  raise SystemExit(rc)

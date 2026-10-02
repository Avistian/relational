"""Reserve build and worker costs before starting Modal; no implicit retries."""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
from relkit.zero_shot_l175 import reserve_cost
R=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',required=True);a=p.parse_args()
 if not a.phase.startswith('audit-') or not a.phase[6:].isdigit():raise ValueError('Use audit-N')
 ledger=R/'labs/evidence/l175/cloud-budget.json'
 with ledger.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);state=json.load(f)
  if any(x['phase']==a.phase for x in state['reservations']):raise ValueError('Attempt already reserved')
  build=state['build_reservation_usd']+1
  total=reserve_cost([build]+[x['upper_usd'] for x in state['reservations']],900,.00006172,2,8)
  state['build_reservation_usd']=build
  state['reservations'].append(dict(phase=a.phase,timeout=900,rate=.00006172,upper_usd=930*.00006172,source_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['modal/l175_repro.py','labs/_audit_context_l175.py','labs/relkit/zero_shot_l175.py']}))
  f.seek(0);json.dump(state,f,indent=2);f.truncate()
 log=R/'labs/evidence/l175'/f'{a.phase}-dispatch.log'
 with log.open('w') as output:
  proc=subprocess.Popen([str(R/'.venv/bin/modal'),'run','modal/l175_repro.py','--phase',a.phase],cwd=R,start_new_session=True,stdout=output,stderr=subprocess.STDOUT)
  try:code=proc.wait(timeout=2100)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait();output.flush()
   app_ids=set(re.findall(r'https://modal.com/apps/[^/]+/[^/]+/(ap-[A-Za-z0-9]+)',log.read_text()))
   for app_id in app_ids:subprocess.run([str(R/'.venv/bin/modal'),'app','stop','--yes',app_id],check=True)
   raise RuntimeError('Build+audit timeout; named remote app stopped; reservation retained')
 print(log.read_text())
 raise SystemExit(code)

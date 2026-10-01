"""Aggregate one-hour local CPU limit; preserve all failures and source identity."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l155/fe'
assert json.loads((E/'sql-audit.json').read_text())['status']=='PASS'
b=json.loads((P/'_budget_l155.json').read_text())
for p,h in b['source_hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
started=E/'fit-started.json';assert not started.exists(),'Single-use author dispatch; use a new namespace for independent repetitions'
started.write_text(json.dumps(dict(started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),limit_seconds=3600,cloud_usd=0)))
start=time.monotonic()
for seed in range(5):
    subprocess.run([sys.executable,str(P/'_run_fe_l155.py'),'--seed',str(seed),'--output',str(E/f'paper/seed-{seed}')],check=True,timeout=max(1,3600-(time.monotonic()-start)),env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
(E/'fit-completed.json').write_text(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-start,searches=5,trials=50,cloud_usd=0),indent=2))

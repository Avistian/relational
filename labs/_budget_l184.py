"""Aggregate local numerical accounting, including failed child commands."""
import json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;f=P/'evidence/l184/budget.json'
b=json.loads(f.read_text()) if f.exists() else dict(cloud_usd=0,aggregate_limit_usd=10,planned_stop_usd=8,reserve_usd=2,local_limit_seconds=3600,attempts=[dict(name='initial inspection and missing-module RED test',seconds=30.,returncode=1,accounting='conservative allowance')])
used=sum(x['seconds'] for x in b['attempts']);remaining=b['local_limit_seconds']-used
if remaining<=0:raise SystemExit('INCOMPLETE_BUDGET_GATE')
start=time.monotonic()
try:r=subprocess.run(sys.argv[1:],timeout=remaining);rc=r.returncode
except subprocess.TimeoutExpired:rc=124
b['attempts'].append(dict(name=' '.join(sys.argv[1:]),seconds=time.monotonic()-start,returncode=rc));b['local_seconds']=sum(x['seconds'] for x in b['attempts'])
f.write_text(json.dumps(b,indent=2)+'\n');raise SystemExit(rc)

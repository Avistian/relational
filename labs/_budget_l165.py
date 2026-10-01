"""Run one sequential author preparation/check command within the aggregate allowance."""
import json,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent/'evidence/l165/budget.json'
r=json.loads(p.read_text()) if p.exists() else dict(cap_seconds=1800,cloud_spend_usd=0,events=[])
remaining=r['cap_seconds']-sum(e['seconds'] for e in r['events'])
if remaining<1:raise SystemExit('INCOMPLETE_BUDGET_GATE')
start=time.monotonic()
try:
 result=subprocess.run(sys.argv[1:],timeout=remaining);status=result.returncode
except subprocess.TimeoutExpired:status=124
r['events'].append(dict(command=sys.argv[1:],seconds=round(time.monotonic()-start,3),exit_code=status))
p.write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(status)

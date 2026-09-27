"""Open the full-run gate only after completed pilot and conservative feasibility."""
import json
from pathlib import Path
P=Path(__file__).parent;path=P/'_budget_l137.json';b=json.loads(path.read_text())
r=json.loads((P/'evidence/l137/pilot/lr005-full/seed-999/completed.json').read_text())
assert r['status']=='COMPLETE' and r['epochs']==1
projection=r['seconds']*10*1.5
assert projection<900 and projection*5*b['rate_usd_second']+3<10
report=dict(status='PASS',one_epoch_worker_seconds=r['seconds'],conservative_ten_epoch_seconds=projection,projected_5_full_fits_usd=projection*5*b['rate_usd_second'],all_fit_slots_reserved_max_usd=6*900*b['rate_usd_second'])
(P/'_pilot_l137_results.json').write_text(json.dumps(report,indent=2));b['pilot_passed']=True;path.write_text(json.dumps(b,indent=2));print(report)

import json
from pathlib import Path
P=Path(__file__).parent;r=json.loads((P/'evidence/l134/pilot/seed-100/completed.json').read_text());a=json.loads((P/'evidence/l134/pilot/seed-100/temporal-audit.json').read_text())
assert r['status']=='COMPLETE' and a['splits']['train']['queries']==7453
projection=5*r['seconds']*10*.00022572
assert projection<1.6
p=P/'_budget_l134.json';b=json.loads(p.read_text());b['pilot_approved_for_full']=True;b['pilot_projection_usd']=projection;p.write_text(json.dumps(b,indent=2))
out=dict(status='PASS',conservative_five_fit_projection_usd=projection,pilot_seconds=r['seconds']);(P/'_pilot_l134_results.json').write_text(json.dumps(out,indent=2));print(out)

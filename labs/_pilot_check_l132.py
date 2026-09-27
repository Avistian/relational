"""Admit all ten fits only after BOTH complete one-epoch pilots fit runtime and cost."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent;B=P/'_budget_l132.json';b=json.loads(B.read_text());rows=[]
for variant in ['sage','idgnn']:
 root=P/f'evidence/l132/pilot/{variant}-100';r=json.loads((root/'result.json').read_text());done=json.loads((root/'completed.json').read_text())
 assert done['status']=='COMPLETE' and r['epochs']==1 and len(r['trace'])==1
 epoch=r['trace'][0]['seconds'];overhead=max(0,r['seconds']-epoch)
 rows.append(dict(variant=variant,epoch_seconds=epoch,projected_seconds=1.25*(overhead+20*epoch)))
reserved=sum(x['reserved_seconds'] for x in b['reservations']);available=b['worker_reserved_seconds']-reserved
approved=all(x['projected_seconds']<=1800 for x in rows) and available>=18000
b.update(pilot_approved_for_full=approved,pilot_projection=rows,projected_ten_fit_worker_usd=sum(x['projected_seconds'] for x in rows)*5*b['rate_usd_second'])
B.write_text(json.dumps(b,indent=2));print(json.dumps(dict(approved=approved,projections=rows),indent=2))

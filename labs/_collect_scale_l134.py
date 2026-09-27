"""Preserve failed attempts and collect only compact successful scale evidence."""
import argparse,json
from pathlib import Path
import modal
P=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--attempt',default='attempt-6');p.add_argument('--failed',action='store_true');a=p.parse_args()
v=modal.Volume.from_name('l134-scale-evidence');out=P/'evidence/l134/scale'/a.attempt;out.mkdir(parents=True,exist_ok=True)
for name in (['failure.json','cost.json','started.json'] if a.failed else ['scale.json','queries.json','cost.json','started.json']):
 raw=b''.join(v.read_file(a.attempt+'/'+name));(out/name).write_bytes(raw)
 if name=='scale.json':
  r=json.loads(raw);assert r['status']=='COMPLETE';(out.parent/name).write_bytes(raw)
print('Collected',a.attempt)

"""Collect every paid-call timer, including failed preflight and pilots; starts no compute."""
import hashlib,json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent
v=modal.Volume.from_name('l107-snapshot-evidence')
out=P/'evidence/l107/calls';out.mkdir(parents=True,exist_ok=True)
records=[]
for entry in v.iterdir('/',recursive=True):
 if entry.type.name!='FILE':continue
 name=Path(entry.path).name
 if not (name.startswith('call-') or name.startswith('replay-call')):continue
 raw=b''.join(v.read_file(entry.path));record=json.loads(raw)
 key=hashlib.sha256(entry.path.encode()).hexdigest()[:12]+'-'+name
 (out/key).write_bytes(raw)
 records.append({'volume_path':entry.path,'file':key,**record})
budget=json.loads((P/'_budget_l107.json').read_text())
seconds=sum(x['seconds'] for x in records)
budget['observed_calls']=records
budget['observed_call_seconds']=seconds
budget['observed_resource_estimate_usd']=seconds*budget['resource_ceiling_usd_second']
budget['estimate_boundary']='Timer-based conservative resource estimate, not provider invoice. Container startup, idle, image builds and storage are excluded; reserved overhead covers these separately.'
assert budget['observed_resource_estimate_usd']<budget['user_limit_usd']
(P/'_budget_l107.json').write_text(json.dumps(budget,indent=2)+'\n')
print({'calls':len(records),'seconds':seconds,'resource_estimate_usd':budget['observed_resource_estimate_usd']})

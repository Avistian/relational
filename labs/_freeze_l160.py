"""Freeze the inherited experiment and replay implementations without rewriting history."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l160';E.mkdir(exist_ok=True)
up=json.loads((P/'evidence/l158/input-manifest.json').read_text())
files=set(up['files'])|{'evidence/l158/input-manifest.json','evidence/l158/report.json','_replay_l158.py','relkit/synthesis_l158.py','_verify_l158.py','_check_l158.py'}
for n,h in up['files'].items():
    if hashlib.sha256((P/n).read_bytes()).hexdigest()!=h:raise ValueError('Inherited input changed: '+n)
m=dict(experiment='L160 Year 4 exit evidence replay',freeze_date='2026-10-01',files={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in sorted(files)})
f=E/'input-manifest.json'
if f.exists():
    if json.loads(f.read_text())!=m:raise ValueError('Frozen inputs changed; do not overwrite')
else:f.write_text(json.dumps(m,indent=2)+'\n')
print('Frozen',len(files),'inputs')

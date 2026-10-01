"""Collect small artifacts only; completed checkpoints remain on named cloud volume."""
import json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;E=P/'evidence/l153';E.mkdir(parents=True,exist_ok=True)
v=modal.Volume.from_name('l153-recommendation-evidence');count=0
for item in v.iterdir('/',recursive=True):
 name=item.path.lstrip('/')
 if name.startswith('cache/') or '/materialized/' in name or '/cache/' in name:continue
 if not name.endswith(('.json','.npz','.txt','.gz','.py')):continue
 dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b''.join(v.read_file(name)));count+=1
print('Collected',count,'small artifacts')

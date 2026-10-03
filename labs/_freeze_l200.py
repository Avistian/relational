"""Seal collected raw inputs after independent pilot admission and full execution."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l200'
paths=[]
for folder in ['packet','pilot-1','full-1']:paths.extend(p for p in (E/folder).rglob('*') if p.is_file())
paths.extend(E/n for n in ['input-manifest.json','preflight.json','admission.json','remote-pilot.json','remote-full.json','pfn-parity.json'])
manifest=dict(files={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},input_manifest_sha256=json.loads((E/'remote-pilot.json').read_text())['input_manifest_sha256'])
p=E/'packet-manifest.json'
if p.exists():assert json.loads(p.read_text())==manifest,'Frozen evidence changed'
else:p.write_text(json.dumps(manifest,indent=2)+'\n')
print('Frozen',len(paths),'inputs')

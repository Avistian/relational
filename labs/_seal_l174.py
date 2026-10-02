"""Seal completed L174 inputs and numerical artifacts without the mutable budget."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l174'
assert json.loads((E/'runs/verification.json').read_text())['status']=='PASS'
files={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.rglob('*')) if p.is_file() and p.name not in {'artifact-manifest.json','local-budget.json'}}
for p in sorted((P/'sources/l174').iterdir()):
 if p.is_file():files[str(Path('../../sources/l174')/p.name)]=hashlib.sha256(p.read_bytes()).hexdigest()
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=files,scope='complete authenticated input and numerical artifacts; excludes mutable budget',whole_paper='NOT_RUN'),indent=2)+'\n')
print('Sealed',len(files),'artifacts')

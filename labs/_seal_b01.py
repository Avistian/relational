"""Seal B01 publication inputs; exclude mutable budget and verification receipts."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b01'
paths=set()
for pattern in ['labs/*b01*.py','labs/b01-*.md','labs/b01-*.ipynb','labs/solutions/b01-*.ipynb','labs/html/b01-*.html','labs/relkit/*b01*.py','lessons/b01-*.html','lessons/content/b01-*.md','assets/comparison-contract.*']:
 paths.update(R.glob(pattern))
for directory in ['labs/evidence/b01','labs/figures/b01']:
 paths.update(p for p in (R/directory).rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json'])
for name in ['reference/b01-comparison-contract.html','reference/curriculum.html','assets/home.js','assets/notebooks.js','lessons/manifest.json','index.html','notebooks.html','labs/README.md','plan/year-5-6-bridge.md']:
 paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n')
print('Sealed',len(paths),'publication inputs')

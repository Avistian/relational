"""Seal publishable B02 artifacts, excluding mutable runtime receipts."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b02';paths=set()
for pattern in ['labs/*b02*.py','labs/b02-*.md','labs/b02-*.ipynb','labs/solutions/b02-*.ipynb','labs/html/b02-*.html','labs/relkit/*b02*.py','lessons/b02-*.html','assets/embedding-ensemble.*','labs/figures/b02/*','modal/b02_repro.py']:
 paths.update(R.glob(pattern))
for p in (E/'compact').rglob('*'):
 if p.is_file():paths.add(p)
for name in ['report.json','source-lock.json','mechanism-audit.json','admission.json','raw-manifest.json','replay.zip','source.zip','budget.json','cost-receipt.json','pilot-receipt.json','full-receipt.json','audited-receipt.json']:
 paths.add(E/name)
for name in ['reference/b02-embeddings-ensembles.html','reference/curriculum.html','assets/home.js','assets/notebooks.js','lessons/manifest.json','index.html','notebooks.html','labs/README.md','plan/year-5-6-bridge.md']:
 paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n');print('Sealed',len(paths),'files')

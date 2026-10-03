"""Seal site-bound B07 artifacts, leaving mutable delivery/budget receipts separate."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b07';paths=set()
for pattern in ['labs/*b07*.py','labs/b07-*.md','labs/b07-*.ipynb','labs/solutions/b07-*.ipynb','labs/html/b07-*.html','labs/relkit/*b07*.py','lessons/b07-*.html','lessons/content/b07-*.md','assets/semantic-transfer.*','labs/figures/b07/*','labs/sources/b07/**/*','labs/data/b07/*']:
 paths.update(p for p in R.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
for p in E.rglob('*'):
 if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json']:paths.add(p)
for name in ['reference/b07-semantic-transfer.html','reference/curriculum.html','reference/glossary.html','lessons/manifest.json','labs/README.md','plan/year-5-6-bridge.md','index.html','notebooks.html','assets/home.js','assets/notebooks.js','assets/retrieval-pool.js','assets/paper-deck.js']:paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n');print('Sealed',len(paths),'files')

"""Seal B06 publication artifacts; mutable delivery receipts and budget kept separate."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b06';paths=set()
for pattern in ['labs/*b06*.py','labs/b06-*.md','labs/b06-*.ipynb','labs/solutions/b06-*.ipynb','labs/html/b06-*.html','labs/relkit/*b06*.py','lessons/b06-*.html','lessons/content/b06-*.md','assets/prior-mixtures.*','labs/figures/b06/*','labs/sources/b06/**/*']:
 paths.update(p for p in R.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
for p in E.rglob('*'):
 if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json']:paths.add(p)
for name in ['reference/b06-prior-mixtures.html','reference/curriculum.html','reference/glossary.html','lessons/manifest.json','labs/README.md','plan/year-5-6-bridge.md','index.html','notebooks.html','assets/home.js','assets/notebooks.js','assets/retrieval-pool.js','assets/paper-deck.js']:paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n');print('Sealed',len(paths),'files')

"""Seal B05 publication artifacts; execution receipts and budget tracked separately."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b05';paths=set()
for pattern in ['labs/*b05*.py','labs/b05-*.md','labs/b05-*.ipynb','labs/solutions/b05-*.ipynb','labs/html/b05-*.html','labs/relkit/*b05*.py','lessons/b05-*.html','lessons/content/b05-*.md','assets/retrieval-episodes.*','labs/figures/b05/*','labs/sources/b05/**/*']:
 paths.update(p for p in R.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
for p in E.glob('*'):
 if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json']:paths.add(p)
for name in ['reference/b05-retrieval-episodes.html','reference/curriculum.html','lessons/manifest.json','labs/README.md','plan/year-5-6-bridge.md','index.html','notebooks.html','assets/home.js','assets/notebooks.js']:paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n');print('Sealed',len(paths),'files')

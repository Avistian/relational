"""Seal B04 publication artifacts; execution receipts and budget tracked separately."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b04';paths=set()
for pattern in ['labs/*b04*.py','labs/b04-*.md','labs/b04-*.ipynb','labs/solutions/b04-*.ipynb','labs/html/b04-*.html','labs/relkit/*b04*.py','lessons/b04-*.html','assets/scalable-icl.*','labs/figures/b04/*','labs/sources/b04/**/*']:
 paths.update(p for p in R.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
for p in E.glob('*'):
 if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json']:paths.add(p)
for name in ['reference/b04-scalable-icl.html','reference/curriculum.html','lessons/manifest.json','labs/README.md','plan/year-5-6-bridge.md']:paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n');print('Sealed',len(paths),'files')

"""Seal publication files; exclude source exploration cache and changing receipts."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b04a';paths=set()
for pattern in ['labs/*b04a*.py','labs/b04a-*.md','labs/b04a-*.ipynb','labs/solutions/b04a-*.ipynb','labs/html/b04a-*.html','labs/relkit/*b04a*.py','lessons/b04a-*.html','lessons/content/b04a-*.md','assets/scaling-axes.*','labs/figures/b04a/*']:
 paths.update(p for p in R.glob(pattern) if p.is_file())
for p in (R/'labs/sources/b04a').glob('*'):
 if p.is_file() and p.name!='upstream.tar.gz' and not p.name.startswith('.'):paths.add(p)
for p in E.rglob('*'):
 if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json']:paths.add(p)
for name in ['reference/b04a-scaling-axes.html','reference/curriculum.html','lessons/manifest.json','labs/README.md','plan/year-5-6-bridge.md','index.html','notebooks.html','assets/home.js','assets/notebooks.js']:paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n');print('Sealed',len(paths),'files')

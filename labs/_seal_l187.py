"""Seal lesson-owned scientific and delivery artifacts after numerical work."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
paths=[R/'lessons/0187-ethics-privacy-reg.html',R/'lessons/content/0187-ethics-privacy-reg.md',
       R/'reference/ethics-privacy-reg.html',R/'assets/entity-privacy.css',R/'assets/entity-privacy.js',R/'assets/l187-lesson.js',
       P/'0187-ethics-privacy-reg.ipynb',P/'solutions/0187-ethics-privacy-reg.ipynb',P/'html/0187-ethics-privacy-reg.html',
       P/'relkit/privacy_l187.py',P/'l187-reproduction.md']
paths+=list(P.glob('_*l187*.py'))+list(P.glob('_*l187*.json'))
for directory in [P/'sources/l187',P/'figures/l187',P/'evidence/l187']:
    paths.extend(x for x in directory.rglob('*') if x.is_file())
paths=sorted(set(p for p in paths if p.name not in ['artifact-manifest.json','_checkout_l187_results.json'] and '__pycache__' not in p.parts))
out={'experiment':'L187-F1-ENTITY-PRIVACY','files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(P/'evidence/l187/artifact-manifest.json').write_text(json.dumps(out,indent=2)+'\n');print('Sealed',len(paths),'files')

"""Seal immutable L186 delivery files; mutable runtime/check-out receipts are excluded."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/l186'
paths=[R/'lessons/0186-production-constraints.html',R/'lessons/content/0186-production-constraints.md',R/'reference/production-constraints.html',P/'0186-production-constraints.ipynb',P/'solutions/0186-production-constraints.ipynb',P/'html/0186-production-constraints.html',P/'relkit/serving_l186.py',P/'l186-reproduction.md']
paths+=list((R/'assets').glob('*l186*'))+[R/'assets/serving-contract.css',R/'assets/serving-contract.js']
paths+=list(P.glob('_*l186*.py'))+list(P.glob('_*l186*results.json'))
for directory in [E,P/'sources/l186',P/'figures/l186']:
 paths.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
exclude={'artifact-manifest.json','local-budget.json','_checkout_l186_results.json'}
files={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.name not in exclude}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=files,exclusions='Mutable local budget and checkout receipt; shared cross-lesson files are not frozen by L186'),indent=2)+'\n')
print('Sealed',len(files),'L186 artifacts')

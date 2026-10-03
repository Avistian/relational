"""Seal publication files; mutable budget and verification receipts stay separate."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0199-select-primary-direction';E=P/'evidence/l199'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/select-primary-direction.html',R/'assets/direction-selection.css',R/'assets/direction-selection.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/direction_l199.py',P/'l199-reproduction.md']
paths+=list(P.glob('_*l199*.py'))+list((P/'figures/l199').glob('*'))
paths+=[p for p in E.rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json'] and '__pycache__' not in str(p)]
r=dict(files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},scope='Publication bytes; mutable budget and result receipts separate')
(E/'artifact-manifest.json').write_text(json.dumps(r,indent=2)+'\n');print('Sealed',len(r['files']),'files')

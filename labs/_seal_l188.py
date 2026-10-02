"""Seal fixed L188 deliverables; mutable budget/check-out receipt stay separate."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0188-systematic-literature-tracking'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/systematic-literature-tracking.html',R/'assets/literature-triage.css',R/'assets/literature-triage.js',R/'assets/l188-lesson.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/literature_l188.py',P/'l188-reproduction.md']
paths+=list(P.glob('_*l188*.py'))+list(P.glob('_*l188*.json'))
for directory in ['evidence/l188','sources/l188','figures/l188']:paths+=list((P/directory).rglob('*'))
excluded={'artifact-manifest.json','local-budget.json','_checkout_l188_results.json'}
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name not in excluded}
(P/'evidence/l188/artifact-manifest.json').write_text(json.dumps(dict(files=files,excluded_mutable=sorted(excluded)),indent=2)+'\n')
print('Sealed',len(files),'files')

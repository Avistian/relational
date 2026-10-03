"""Authenticate fixed L191 deliverables; exclude mutable budget/build receipts."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0191-kumorfm2-sota-tracking'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/kumorfm2-sota-tracking.html',R/'assets/sota-tracking.css',R/'assets/sota-tracking.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/tracking_l191.py',P/'l191-reproduction.md']
paths+=list(P.glob('_*l191*.py'))+list(P.glob('_*l191*.json'))
for directory in ['evidence/l191','figures/l191','sources/l191']:paths+=list((P/directory).rglob('*'))
excluded={'artifact-manifest.json','local-budget.json','_checkout_l191_results.json'}
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name not in excluded}
(P/'evidence/l191/artifact-manifest.json').write_text(json.dumps(dict(files=files,excluded_mutable=sorted(excluded)),indent=2)+'\n');print('Sealed',len(files),'fixed files')

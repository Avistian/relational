"""Seal fixed L190 deliverables; mutable runtime and publication receipt excluded."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0190-research-gap-checkpoint'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'lessons/content/0190-research-gap-document.md',R/'reference/research-gap-checkpoint.html',R/'reference/research-gap-document.html',R/'reference/research-gap-document.pdf',R/'assets/research-checkpoint.css',R/'assets/research-checkpoint.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/checkpoint_l190.py',P/'l190-reproduction.md']
paths+=list(P.glob('_*l190*.py'))+list(P.glob('_*l190*.json'))
for directory in ['evidence/l190','figures/l190']:paths+=list((P/directory).rglob('*'))
excluded={'artifact-manifest.json','local-budget.json','_checkout_l190_results.json'}
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name not in excluded}
(P/'evidence/l190/artifact-manifest.json').write_text(json.dumps(dict(files=files,excluded_mutable=sorted(excluded)),indent=2)+'\n')
print('Sealed',len(files),'fixed files')

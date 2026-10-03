"""Seal immutable L193 delivery; budget and final publication receipt stay mutable."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';S='0193-open-fm-full-task-set'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/open-fm-full-task-set.html',R/'assets/multitask-reproduction.css',R/'assets/multitask-reproduction.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/multitask_l193.py',P/'l193-reproduction.md']
paths+=list(P.glob('_*l193*.py'))+list(P.glob('_*l193*.json'))
for directory in ['evidence/l193','figures/l193','sources/l193']:paths+=list((P/directory).rglob('*'))
excluded={'artifact-manifest.json','local-budget.json','_checkout_l193_results.json'}
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts and p.suffix!='.pyc'}
(P/'evidence/l193/artifact-manifest.json').write_text(json.dumps(dict(files=files,excluded_mutable=sorted(excluded)),indent=2)+'\n');print('Sealed',len(files),'fixed files')

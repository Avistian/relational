"""Seal authored deliverables and authenticated input packet, not this manifest."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1];P=R/'labs'
paths=[R/'lessons/0184-gelgt-temporal-attention.html',R/'lessons/content/0184-gelgt-temporal-attention.md',R/'reference/gelgt-temporal-attention.html',R/'assets/gelgt-viz.js',R/'assets/gelgt-viz.css',R/'assets/l184-lesson.js',P/'0184-gelgt-temporal-attention.ipynb',P/'solutions/0184-gelgt-temporal-attention.ipynb',P/'html/0184-gelgt-temporal-attention.html',P/'relkit/gelgt_l184.py',P/'l184-reproduction.md',R/'modal/l184_paper_repro.py']
paths+=list(P.glob('_*l184*.py'))+list(P.glob('_*l184*.json'))
for name in ['evidence/l184','sources/l184','figures/l184']:paths+=list((P/name).rglob('*'))
paths=[p for p in paths if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact-manifest.json','_checkout_l184_results.json']]
r={'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}}
(P/'evidence/l184/artifact-manifest.json').write_text(json.dumps(r,indent=2)+'\n');print(len(r['files']),'sealed files')

"""Seal deliverables; exclude mutable accounting and the seal itself."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='b19a-predictive-distributions'
paths=[*P.glob('_*b19a*.py'),*P.glob('_*b19a*.json'),P/'relkit/scoring_b19a.py',P/'b19a-reproduction.md',P/'b19a-metric-contract.md',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html',R/'lessons'/f'{S}.html',R/'lessons/content'/f'{S}.md',R/'reference'/f'{S}.html',R/'assets/b19a-evidence.js',R/'assets/distribution-scores.css',R/'assets/distribution-scores.js',R/'docs/plans/2026-10-04-b19a-design.md',R/'learning-records/0176-predictive-distributions-prepared.md']
for folder in ['sources/b19a','evidence/b19a','figures/b19a']:paths.extend((P/folder).rglob('*'))
paths=[p for p in paths if p.is_file() and '__pycache__' not in p.parts and p.name not in ['seal.json','local-budget.json','_pages_b19a_results.json']]
manifest=[dict(file=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths))]
(P/'evidence/b19a/seal.json').write_text(json.dumps(dict(files=manifest,exclusions=['mutable local-budget.json','self seal.json','self-referential Pages result']),indent=2)+'\n')
for x in manifest:assert hashlib.sha256((R/x['file']).read_bytes()).hexdigest()==x['sha256']
print('Sealed',len(manifest),'files',sum(x['bytes'] for x in manifest),'bytes')

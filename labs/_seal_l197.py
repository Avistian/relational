"""Seal deliverable bytes; mutable execution receipts and budget stay separate."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0197-year-5-essay';E=P/'evidence/l197'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/year-5-essay.html',R/'assets/arch-family-viz.js',R/'assets/landscape-essay.css',R/'assets/landscape-essay.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/landscape_l197.py',P/'l197-reproduction.md']
paths+=list(P.glob('_*l197*.py'))+list((P/'figures/l197').glob('*'))
paths += [p for p in E.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact-manifest.json','local-budget.json']]
result=dict(files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},scope='Publication bytes; budget and execution receipts separate')
(E/'artifact-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('Sealed',len(result['files']),'files')

"""Seal deliverable bytes; mutable execution receipts and budget stay separate."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0198-three-research-directions';E=P/'evidence/l198'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/three-research-directions.html',R/'assets/research-priority.js',R/'assets/research-priority.css',R/'assets/proposal-decisions.css',R/'assets/proposal-decisions.js',R/'assets/l198-lesson.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/proposals_l198.py',P/'l198-reproduction.md']
paths+=list(P.glob('_*l198*.py'))+list((P/'figures/l198').glob('*'))
paths += [p for p in E.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact-manifest.json','local-budget.json']]
result=dict(files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},scope='Publication bytes; budget and execution receipts separate')
(E/'artifact-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('Sealed',len(result['files']),'files')

"""Seal intended publication bytes; mutable budget and check receipts stay separate."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0200-year-5-exit-exam';E=P/'evidence/l200'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/year-5-exit-exam.html',R/'assets/year-five-exit.css',R/'assets/year-five-exit.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/exit_l200.py',P/'l200-reproduction.md',R/'modal/l200_repro.py']
paths+=list(P.glob('_*l200*.py'))+list((P/'figures/l200').glob('*'))+list((P/'sources/l200').glob('*'))
paths+=[p for p in E.rglob('*') if p.is_file() and p.name not in ['artifact-manifest.json','local-budget.json'] and '__pycache__' not in str(p)]
r=dict(files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},scope='Publication bytes; mutable local budget and verification receipts separate')
(E/'artifact-manifest.json').write_text(json.dumps(r,indent=2)+'\n');print('Sealed',len(r['files']),'files')

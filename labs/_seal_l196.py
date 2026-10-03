"""Seal published deliverables; retain mutable receipts and budget separately."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0196-community-engagement';E=P/'evidence/l196'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/community-engagement.html',R/'assets/community-question.css',R/'assets/community-question.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/community_l196.py',P/'l196-reproduction.md']
paths+=list(P.glob('_*l196*.py'))
paths += [p for p in E.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact-manifest.json','local-budget.json']]
result=dict(files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},scope='Publication bytes; budget and check receipts separate')
(E/'artifact-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('Sealed',len(result['files']),'files')

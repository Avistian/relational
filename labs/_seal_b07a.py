"""Seal deliverable bytes, excluding5GB external checkpoint and volatile budget/logs."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b07a';S='b07a-hypernetworks'
files=[]
for root in [P/'sources/b07a',P/'figures/b07a',P/'data/b07a',E/'runs']:
 files += [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.ckpt','.partial'] and p.name!='partial.json']
files += list(P.glob('_*b07a.py'))+list((P/'relkit').glob('*b07a.py'))
files += [R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/b07a-hypernetworks.html',R/'assets/hypernetworks.css',R/'assets/hypernetworks.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'b07a-reproduction.md']
files += [E/n for n in ['course-protocol.json','source-gate.json','course-audit.json','source-parity.json','reproducer.zip','solution-report.json','execution-environment.json']]
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest,checkpoint='External authenticated5.09GB; deliberately excluded',learner='PENDING_WRITTEN_DEFENSE'),indent=2)+'\n')
print('Sealed',len(manifest),'files')

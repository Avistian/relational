"""Authenticate portable B08 artifacts; keep volatile budget/report out of seal."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b08';S='b08-structured-objectives'
files=[]
for root in [P/'sources/b08',P/'figures/b08',P/'data/b08',E/'runs']:
 files += [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.ckpt','.gz']]
files += list(P.glob('_*b08.py'))+list((P/'relkit').glob('*b08.py'))
files += [R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference'/(S+'.html'),R/'assets/structured-objectives.css',R/'assets/structured-objectives.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'b08-reproduction.md']
files += [E/n for n in ['course-protocol.json','source-gate.json','course-audit.json','reproducer.zip','solution-report.json']]
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest,checkpoint='Metadata pinned; no weight download or inference',learner='PENDING_WRITTEN_DEFENSE'),indent=2)+'\n');print('Sealed',len(manifest),'files')

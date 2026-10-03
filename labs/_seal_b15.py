"""Seal stable B15 artifacts; exclude mutable accounting and post-seal reports."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';paths=[]
for root in [P/'sources/b15',P/'evidence/b15',P/'figures/b15']:
 paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact-manifest.json','local-budget.json','cloud-budget.json','deployment.json','publication-files.json'])
for pattern in ['_*b15*.py','b15-*','solutions/b15-*','html/b15-*','relkit/labels_b15.py']:paths+=list(P.glob(pattern))
for name in ['lessons/b15-parameter-free-encoders.html','lessons/content/b15-parameter-free-encoders.md','reference/b15-parameter-free-encoders.html','assets/label-visibility.js','assets/label-visibility.css','docs/plans/2026-10-04-b15-design.md','learning-records/0171-parameter-free-encoders-prepared.md','reviews/lesson-b15/review.md']:paths.append(R/name)
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}
(P/'evidence/b15/artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Sealed',len(manifest),'files')

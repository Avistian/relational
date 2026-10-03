"""Seal distributable B14 files, excluding mutable budget and post-seal reports."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';paths=[]
for root in [P/'sources/b14',P/'evidence/b14',P/'figures/b14']:
 paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact-manifest.json','local-budget.json','cloud-budget.json'])
for pattern in ['_*b14*.py','b14-*','solutions/b14-*','html/b14-*','relkit/flatten_b14.py']:paths+=list(P.glob(pattern))
for name in ['lessons/b14-flattening-challenge.html','lessons/content/b14-flattening-challenge.md','reference/b14-flattening-challenge.html','assets/flattening.js','assets/flattening.css','modal/b14_tabpfn_rel.py','docs/plans/2026-10-03-b14-design.md']:paths.append(R/name)
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}
(P/'evidence/b14/artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Sealed',len(manifest),'files')

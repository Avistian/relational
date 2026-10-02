"""Seal complete pinned inputs, numerical evidence and delivered lesson artifacts."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l178';S='0178-fair-model-comparison'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+[p for p in (P/'sources/l178').rglob('*') if p.is_file() and '__pycache__' not in p.parts]+list((P/'figures/l178').iterdir())
paths+=[P/'relkit/comparison_l178.py',P/'l178-reproduction.md',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/fair-model-comparison.html']
paths += [P/'_verify_l178_results.json',P/'_execution_l178_results.json',P/'_delivery_l178_results.json']
paths+=list(P.glob('_*l178*.py'))+[R/'assets/comparison-gate.css',R/'assets/comparison-gate.js',R/'assets/l178-lesson.js']
pins=json.loads((E/'audit-input-manifest.json').read_text())['files']
for name,digest in pins.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
paths += [P/name for name in pins]
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Sealed',len(manifest),'files')

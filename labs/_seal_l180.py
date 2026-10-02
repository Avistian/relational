"""Seal complete pinned inputs, numerical evidence and delivered lesson artifacts."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l180';S='0180-public-encoder-checkpoint'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+[p for p in (P/'sources/l180').rglob('*') if p.is_file() and '__pycache__' not in p.parts]+list((P/'figures/l180').iterdir())
paths+=[P/'relkit/checkpoint_l180.py',P/'l180-reproduction.md',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/public-encoder-checkpoint.html']
paths += [P/'_verify_l180_results.json',P/'_execution_l180_results.json',P/'_delivery_l180_results.json']
paths+=list(P.glob('_*l180*.py'))+[R/'assets/encoder-checkpoint.css',R/'assets/encoder-checkpoint.js',R/'assets/l180-lesson.js']
pins=json.loads((E/'input-manifest.json').read_text())['files']
for name,digest in pins.items():assert hashlib.sha256((E/'packet'/name).read_bytes()).hexdigest()==digest,name
inherited=json.loads((E/'input-manifest.json').read_text())['inherited_sealed_files']
for name,digest in inherited.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Sealed',len(manifest),'files; preserved',len(inherited),'inherited inputs')

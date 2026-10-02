"""Seal L183 delivered artifacts and verify inherited evidence remains unchanged."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l183';S='0183-graph-transformer-pretraining'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+[p for p in (P/'sources/l183').rglob('*') if p.is_file()]+list((P/'figures/l183').iterdir())
paths += [P/'relkit/pretraining_l183.py',P/'l183-reproduction.md',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/graph-transformer-pretraining.html']
paths += list(P.glob('_*l183*.py'))+[P/'_check_viz_l183.js',P/'_verify_l183_results.json',P/'_execution_l183_results.json',P/'_delivery_l183_results.json',R/'assets/factorial-contrast.css',R/'assets/factorial-contrast.js',R/'assets/l183-lesson.js']
pins=json.loads((E/'input-manifest.json').read_text())['files']
for name,digest in pins.items():
 for p in [E/'packet'/name,R/name]:assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,str(p)
files={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=files,inherited_inputs=len(pins)),indent=2)+'\n');print('Sealed',len(files),'files; preserved',len(pins),'inherited inputs')

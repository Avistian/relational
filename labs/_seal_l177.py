"""Seal the full input universe and all lesson delivery artifacts."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l177';S='0177-compute-budget-realism'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+sorted((P/'sources/l177').iterdir())+sorted((P/'figures/l177').iterdir())
paths+=[P/'relkit/compute_l177.py',P/'l177-reproduction.md',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/compute-budget-realism.html']
paths += [P/'_verify_l177_results.json',P/'_execution_l177_results.json',P/'_delivery_l177_results.json']
paths+=sorted(P.glob('_*l177*.py'))+sorted(R.glob('assets/*budget*'))+[R/'assets/l177-evidence.js',R/'assets/l177-lesson.js']
paths += [P/name for name in json.loads((E/'input-manifest.json').read_text())['files']]
for name,digest in json.loads((E/'input-manifest.json').read_text())['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Sealed',len(manifest),'files')

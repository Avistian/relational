"""Seal L182 delivery and authenticate all inherited release/model dependencies."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l182';S='0182-rdb-pfn-composite-message-passing'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+[p for p in (P/'sources/l182').rglob('*') if p.is_file() and '__pycache__' not in p.parts]+list((P/'figures/l182').iterdir())
paths += [P/'relkit/composite_l182.py',P/'l182-reproduction.md',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/rdb-pfn-composite-message-passing.html',R/'modal/l182_repro.py']
paths += [P/'_verify_l182_results.json',P/'_execution_l182_results.json',P/'_delivery_l182_results.json']
paths += list(P.glob('_*l182*.py'))+[R/'assets/composite-route.css',R/'assets/composite-route-viz.js',R/'assets/l182-lesson.js']
ledger=json.loads((P/'sources/l182/source-ledger.json').read_text())
for name,h in ledger['inherited_files'].items():
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
 paths.append(R/name)
original=json.loads((P/'_sources_l143.json').read_text())
for name,v in original['files'].items():
 path=P/'sources/l141'/name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==v['sha256'],name
 paths.append(path)
paths.append(P/'_sources_l143.json')
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Sealed',len(manifest),'files; inherited release authentication PASS')

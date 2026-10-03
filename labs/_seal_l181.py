"""Seal complete inputs, numerical evidence and the delivered lesson package."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l181';S='0181-relbench-v2-autocomplete'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+[p for p in (P/'sources/l181').rglob('*') if p.is_file() and '__pycache__' not in p.parts]+list((P/'figures/l181').iterdir())
paths += [P/'relkit/autocomplete_l181.py',P/'l181-reproduction.md',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/relbench-v2-autocomplete.html',R/'modal/l181_paper_repro.py']
paths += [P/'_verify_l181_results.json',P/'_execution_l181_results.json',P/'_delivery_l181_results.json',P/'_admission_l181_results.json']
paths += list(P.glob('_*l181*.py'))+[R/'assets/autocomplete-visibility.css',R/'assets/autocomplete-visibility.js',R/'assets/l181-lesson.js']
for name,digest in json.loads((E/'input-manifest.json').read_text())['files'].items():assert hashlib.sha256((E/'packet'/name).read_bytes()).hexdigest()==digest,name
ledger=json.loads((P/'sources/l181/source-ledger.json').read_text())
for name,digest in ledger['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
for name,digest in ledger['raw_sha256'].items():assert hashlib.sha256((P/'evidence/l171/db'/name).read_bytes()).hexdigest()==digest,name
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');print('Sealed',len(manifest),'files; authenticated all9inherited tables')

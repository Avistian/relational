"""Seal source, execution and evidence files for copied-site verification."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l176';S=P/'sources/l176'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+[p for p in S.rglob('*') if p.is_file()]+[P/'relkit/few_shot_l176.py']+sorted(P.glob('_*l176*.py'))+[R/'modal/l176_repro.py']
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
for name,digest in json.loads((E/'audit-manifest.json').read_text())['files'].items():
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 manifest['labs/'+name]=digest
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
print('Sealed',len(manifest),'files')

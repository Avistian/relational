"""Seal source, execution and evidence files for copied-site verification."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l175';S=P/'sources/l175'
paths=[p for p in E.rglob('*') if p.is_file() and p.name!='artifact-manifest.json']+[p for p in S.rglob('*') if p.is_file()]+[P/'relkit/zero_shot_l175.py']+sorted(P.glob('_*l175*.py'))+[R/'modal/l175_repro.py']
manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
print('Sealed',len(manifest),'files')

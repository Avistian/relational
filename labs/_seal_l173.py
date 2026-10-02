"""Freeze artifact byte identities after validation; budget ledger remains append-only."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l173'
files={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.rglob('*')) if p.is_file() and p.name not in ['local-budget.json','artifact-manifest.json']}
(E/'artifact-manifest.json').write_text(json.dumps(dict(files=files,scope='Input packet, six fits, all epoch checkpoints, predictions, baseline and reports; excludes evolving budget ledger'),indent=2)+'\n')
print('Sealed',len(files),'artifacts')

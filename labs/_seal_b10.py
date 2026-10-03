"""Seal B10 static deliverables; mutable budget and delivery receipts are excluded."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
paths=[]
for pattern in ['labs/_*b10*.py','labs/b10*','labs/relkit/rt_b10.py','labs/sources/b10/**/*','labs/figures/b10/*','labs/html/b10*','labs/solutions/b10*','lessons/b10*','lessons/content/b10*','reference/b10*','assets/relational-cell-attention.*','labs/evidence/b10/mechanism.json','labs/evidence/b10/temporal-audit.json']:
 paths.extend(R.glob(pattern))
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and '__pycache__' not in str(p)}
(P/'evidence/b10/artifact-manifest.json').write_text(json.dumps({'files':files,'status':'SEALED','checkpoint_bytes':'NOT_AUTHENTICATED'},indent=2)+'\n');print('Sealed',len(files),'B10 artifacts')

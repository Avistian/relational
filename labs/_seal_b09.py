"""Seal deliverable bytes; reruns must authenticate rather than overwrite results."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
paths=[]
for pattern in ['labs/_*b09*.py','labs/_*b09*.json','labs/b09*','labs/relkit/cost_b09.py','labs/data/b09/*','labs/evidence/b09/*','labs/sources/b09/*','labs/figures/b09/*','labs/html/b09*','labs/solutions/b09*','lessons/b09*','lessons/content/b09*','reference/b09*','assets/cost-frontier.*']:
 paths.extend(R.glob(pattern))
excluded={'artifact-manifest.json','_checkout_b09_results.json'}
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name not in excluded}
(P/'evidence/b09/artifact-manifest.json').write_text(json.dumps({'files':files,'status':'SEALED','weights':'external pinned identities in sources/b09/weights.json'},indent=2));print('sealed',len(files),'files')

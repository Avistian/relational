"""Seal the complete local B19b delivery; --check verifies every pinned artifact."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b19b';seal=E/'artifact-manifest.json'
if '--check' in sys.argv:
 manifest=json.loads(seal.read_text())
 for row in manifest['files']:
  assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 print('PASS:',len(manifest['files']),'sealed files')
else:
 paths=[]
 for root in [P/'sources/b19b',P/'evidence/b19b',P/'figures/b19b']:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=seal)
 for pattern in ['_*b19b*.py','_*b19b*.json','b19b-*.md','b19b-*.ipynb']:paths.extend(P.glob(pattern))
 paths.extend([P/'relkit/forecast_b19b.py',P/'solutions/b19b-forecasting-contracts.ipynb',P/'html/b19b-forecasting-contracts.html',R/'lessons/b19b-forecasting-contracts.html',R/'lessons/content/b19b-forecasting-contracts.md',R/'reference/b19b-forecasting-contracts.html',R/'assets/forecast-availability.js',R/'assets/forecast-availability.css',R/'docs/plans/2026-10-04-b19b-design.md'])
 files=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(paths))]
 seal.write_text(json.dumps(dict(status='SEALED_LOCAL_DELIVERY',files=files,limits='Hash identity is not historical paper reproduction, deployment or learner mastery'),indent=2)+'\n');print('Sealed',len(files),'files')

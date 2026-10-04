"""Seal immutable B22 package files; accounting and deployment receipts are separate."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b22';seal=E/'artifact-manifest.json';S='b22-support-state-refinement'
if '--check' in sys.argv:
 m=json.loads(seal.read_text())
 for row in m['files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 print('PASS:',len(m['files']),'sealed B22 files')
else:
 paths=[]
 for root in [P/'sources/b22',P/'evidence/b22',P/'figures/b22']:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=seal and p.name!='local-budget.json')
 for pattern in ['_*b22*.py','_*b22*.json','b22-*.md','b22-*.ipynb']:paths.extend(P.glob(pattern))
 paths.extend([P/'relkit/refinement_b22.py',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html',R/'lessons'/f'{S}.html',R/'lessons/content'/f'{S}.md',R/'reference'/f'{S}.html',R/'assets/support-state.css',R/'assets/support-state.js',R/'docs/plans/2026-10-04-b22-design.md',R/'reviews/lesson-b22/README.md'])
 rows=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(paths))]
 seal.write_text(json.dumps(dict(status='SEALED_LOCAL_DELIVERY',files=rows,limits='Historical paper inference NOT_RUN; ledger and subsequent live deployment receipt are separate'),indent=2)+'\n');print('Sealed',len(rows),'B22 files')

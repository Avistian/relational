"""Seal B24 delivery; runtime ledger and later live receipt are separate."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b24';seal=E/'artifact-manifest.json';S='b24-architecture-thesis-defense'
if '--check' in sys.argv:
 m=json.loads(seal.read_text())
 for row in m['files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 print('PASS:',len(m['files']),'sealed B24 files')
else:
 paths=[]
 for root in [P/'sources/b24',E,P/'figures/b24']:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=seal and p.name!='local-budget.json')
 for pattern in ['_*b24*.py','_*b24*.json','b24-*.md','b24-*.ipynb']:paths.extend(P.glob(pattern))
 paths.extend([P/'relkit/defense_b24.py',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html',R/'lessons'/f'{S}.html',R/'lessons/content'/f'{S}.md',R/'reference'/f'{S}.html',R/'reference/b24-proposal-template.html',R/'assets/research-defense.css',R/'assets/research-defense.js',R/'docs/plans/2026-10-04-b24-design.md',R/'reviews/lesson-b24/README.md'])
 rows=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(paths))]
 seal.write_text(json.dumps(dict(status='SEALED_DELIVERY',files=rows,limits='Complete saved-evidence replay; historical identity/availability NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE; no fresh B24 model execution'),indent=2)+'\n');print('Sealed',len(rows),'B24 files')

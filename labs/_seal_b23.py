"""Seal immutable B23 package files; accounting and deployment receipts are separate."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b23';seal=E/'artifact-manifest.json';S='b23-declared-comparison'
if '--check' in sys.argv:
 m=json.loads(seal.read_text())
 for row in m['files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 print('PASS:',len(m['files']),'sealed B23 files')
else:
 paths=[]
 for root in [P/'sources/b23',P/'evidence/b23',P/'figures/b23']:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=seal and p.name!='local-budget.json')
 for pattern in ['_*b23*.py','_*b23*.json','b23-*.md','b23-*.ipynb']:paths.extend(P.glob(pattern))
 paths.extend([P/'relkit/comparison_b23.py',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html',R/'lessons'/f'{S}.html',R/'lessons/content'/f'{S}.md',R/'reference'/f'{S}.html',R/'assets/comparison-evidence.css',R/'assets/comparison-evidence.js',R/'assets/b23-results.js',R/'modal/b23_repro.py',R/'docs/plans/2026-10-04-b23-design.md',R/'reviews/lesson-b23/README.md'])
 rows=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(paths))]
 seal.write_text(json.dumps(dict(status='SEALED_LOCAL_DELIVERY',files=rows,limits='Selected release inference COMPLETE; historical identity NOT_ESTABLISHED; accounting and live receipt are separate'),indent=2)+'\n');print('Sealed',len(rows),'B23 files')

"""Seal the final local package; execution evidence is distinct from paper identity."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b20';seal=E/'artifact-manifest.json'
if '--check' in sys.argv:
 m=json.loads(seal.read_text())
 for row in m['files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 print('PASS:',len(m['files']),'sealed artifacts')
else:
 paths=[]
 for root in [P/'sources/b20',P/'evidence/b20',P/'figures/b20']:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=seal)
 for pattern in ['_*b20*.py','_*b20*.json','b20-*.md','b20-*.ipynb']:paths.extend(P.glob(pattern))
 paths.extend([P/'relkit/curriculum_b20.py',P/'solutions/b20-curriculum-order.ipynb',P/'html/b20-curriculum-order.html',R/'lessons/b20-curriculum-order.html',R/'lessons/content/b20-curriculum-order.md',R/'reference/b20-curriculum-order.html',R/'assets/curriculum-order.css',R/'assets/curriculum-order.js',R/'docs/plans/2026-10-04-b20-design.md',R/'reviews/lesson-b20/README.md'])
 rows=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(paths))]
 seal.write_text(json.dumps(dict(status='SEALED_LOCAL_DELIVERY',files=rows,limits='Local source/artifact identity is not historical paper reproduction, deployment or learner mastery'),indent=2)+'\n');print('Sealed',len(rows),'artifacts')

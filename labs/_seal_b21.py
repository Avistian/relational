"""Seal the final local package; execution evidence is distinct from paper identity."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b21';seal=E/'artifact-manifest.json'
if '--check' in sys.argv:
 m=json.loads(seal.read_text())
 for row in m['files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 print('PASS:',len(m['files']),'sealed artifacts')
else:
 paths=[]
 for root in [P/'sources/b21',P/'evidence/b21',P/'figures/b21']:
  paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=seal)
 for pattern in ['_*b21*.py','_*b21*.json','b21-*.md','b21-*.ipynb']:paths.extend(P.glob(pattern))
 paths.extend([P/'relkit/structural_b21.py',P/'solutions/b21-structural-robustness.ipynb',P/'html/b21-structural-robustness.html',R/'lessons/b21-structural-robustness.html',R/'lessons/content/b21-structural-robustness.md',R/'reference/b21-structural-robustness.html',R/'assets/structural-robustness.css',R/'assets/structural-robustness.js',R/'docs/plans/2026-10-04-b21-design.md',R/'reviews/lesson-b21/README.md'])
 rows=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(paths))]
 seal.write_text(json.dumps(dict(status='SEALED_LOCAL_DELIVERY',files=rows,limits='Local source/artifact identity is not historical paper reproduction, deployment or learner mastery'),indent=2)+'\n');print('Sealed',len(rows),'artifacts')

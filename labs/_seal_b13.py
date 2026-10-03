"""Seal delivered bytes, excluding mutable accounting and copied-build receipt."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b13';S='b13-synthetic-relational-data'
files=[]
for pattern in ['labs/_*b13*.py','labs/_*b13*.json','labs/sources/b13/**/*','labs/figures/b13/*','labs/evidence/b13/*']:
    files += [p for p in R.glob(pattern) if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact-manifest.json','local-budget.json','_checkout_b13_results.json']]
for name in [f'lessons/{S}.html',f'lessons/content/{S}.md',f'reference/{S}.html',f'labs/{S}.ipynb',f'labs/solutions/{S}.ipynb',f'labs/html/{S}.html','labs/b13-reproduction.md','labs/relkit/synthetic_b13.py','assets/synthetic-data.css','assets/synthetic-data.js','docs/plans/2026-10-03-b13-design.md','learning-records/0169-synthetic-relational-data-prepared.md']:
    files.append(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}},indent=2)+'\n');print('Sealed',len(set(files)),'B13 files')

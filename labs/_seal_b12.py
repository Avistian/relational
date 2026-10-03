"""Seal B12 delivery; exclude mutable budget and checkout result to avoid cycles."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b12';S='b12-adaptation-mechanisms'
files=[]
for pattern in ['labs/_*b12*.py','labs/_*b12*.json','labs/sources/b12/**/*','labs/figures/b12/*','labs/evidence/b12/*']:
 files += [p for p in R.glob(pattern) if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact-manifest.json','local-budget.json','_checkout_b12_results.json']]
for name in [f'lessons/{S}.html',f'lessons/content/{S}.md',f'reference/{S}.html',f'labs/{S}.ipynb',f'labs/solutions/{S}.ipynb',f'labs/html/{S}.html','labs/b12-reproduction.md','labs/relkit/adaptation_b12.py','assets/adaptation-paths.css','assets/adaptation-paths.js','docs/plans/2026-10-03-b12-design.md','learning-records/0168-adaptation-mechanisms-prepared.md']:
 files.append(R/name)
result={'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}}
(E/'artifact-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('Sealed',len(result['files']),'B12 files')

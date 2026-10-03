"""Seal delivered B11 artifacts, excluding mutable budget and checkout receipts."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'labs';E=P/'evidence/b11';S='b11-supervised-relational-baselines'
files=[]
for pattern in ['labs/_*b11*.py','labs/_*b11*.json','labs/sources/b11/**/*','labs/figures/b11/*']:
    files += [p for p in R.glob(pattern) if p.is_file() and '__pycache__' not in str(p) and '_checkout_b11_results' not in p.name]
for name in [f'lessons/{S}.html',f'lessons/content/{S}.md',f'reference/{S}.html',f'labs/{S}.ipynb',f'labs/solutions/{S}.ipynb',f'labs/html/{S}.html','labs/b11-reproduction.md','labs/relkit/baselines_b11.py','assets/relational-baselines.css','assets/relational-baselines.js','labs/evidence/b11/mechanism.json','labs/evidence/b11/replay.json','labs/evidence/b11/portable-source-evidence.zip']:
    files.append(R/name)
result={'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}}
(E/'artifact-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('Sealed',len(result['files']),'B11 files')

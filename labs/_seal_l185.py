"""Seal the current L185 package after builds/checks; not a scientific result check."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
patterns=['assets/*l185*','assets/causal-intervention.*','lessons/0185-*','lessons/content/0185-*','reference/causal-relational-data.html','labs/*l185*.py','labs/*l185*.json','labs/l185-*.md','labs/0185-*','labs/solutions/0185-*','labs/html/0185-*','labs/relkit/causal_l185.py','labs/figures/l185/*','labs/sources/l185/*','labs/evidence/l185/**/*','docs/plans/*lesson-185*','learning-records/*causal-relational-data*']
files={p for pattern in patterns for p in R.glob(pattern) if p.is_file() and p.name not in ['budget.lock','package-manifest.json','_checkout_l185_results.json']}
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(R/'labs/evidence/l185/package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Sealed',len(manifest),'L185 files')

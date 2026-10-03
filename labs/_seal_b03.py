"""Seal B03 publication artifacts; runtime receipts are tracked separately."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b03';paths=set()
for pattern in ['labs/*b03*.py','labs/b03-*.md','labs/b03-*.ipynb','labs/solutions/b03-*.ipynb','labs/html/b03-*.html','labs/relkit/*b03*.py','lessons/b03-*.html','assets/pfn-contracts.*','labs/figures/b03/*','labs/sources/b03/*.zip','labs/sources/b03/openml-task-*.json','labs/sources/b03/github-datasets.txt']:
 paths.update(R.glob(pattern))
for name in ['source-gate.json','source-lock.json','permutation-diagnostic.json','diagnostic-audit.json','versions.json','reproducer.zip']:paths.add(E/name)
for name in ['reference/b03-pfn-contracts.html','reference/curriculum.html','lessons/manifest.json','labs/README.md','plan/year-5-6-bridge.md']:paths.add(R/name)
(E/'artifact-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}},indent=2)+'\n');print('Sealed',len(paths),'files')

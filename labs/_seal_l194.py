"""Seal deliverable bytes, excluding mutable budget/check outputs."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0194-open-fm-analysis-report'
paths=[R/'lessons'/(S+'.html'),R/'reference/open-fm-analysis-report.html',R/'assets/reproduction-report.js',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/report_l194.py',P/'l194-reproduction.md']
paths+=list((P/'figures/l194').glob('*'))+list((P/'evidence/l194/packet').glob('*'))+[P/'evidence/l194/input-manifest.json',P/'evidence/l194/report.json',P/'evidence/l194/report.md']
paths+=list(P.glob('_*l194*.py'))
result=dict(files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},scope='Prepared delivery bytes; excludes mutable budget and verification receipts')
(P/'evidence/l194/artifact-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('Sealed',len(result['files']),'files')

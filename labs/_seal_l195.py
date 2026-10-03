"""Seal final delivery bytes; leave budget and check receipts mutable."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0195-thesis-stress-test'
paths=[R/'lessons'/(S+'.html'),R/'lessons/content'/(S+'.md'),R/'reference/thesis-stress-test.html',R/'assets/thesis-stress.js',R/'assets/thesis-stress.css',P/(S+'.ipynb'),P/'solutions'/(S+'.ipynb'),P/'html'/(S+'.html'),P/'relkit/stress_l195.py',P/'l195-reproduction.md']
paths+=list((P/'figures/l195').glob('*'))+[p for p in (P/'evidence/l195/packet').rglob('*') if p.is_file()]+[P/'evidence/l195/input-manifest.json',P/'evidence/l195/report.json',P/'evidence/l195/falsification-brief.md']
paths+=list(P.glob('_*l195*.py'))
r=dict(files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},scope='Prepared delivery bytes; excludes mutable budget and verification receipts')
(P/'evidence/l195/artifact-manifest.json').write_text(json.dumps(r,indent=2)+'\n');print('Sealed',len(r['files']),'files')

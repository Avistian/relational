"""Freeze complete primary evidence without importing weights into the publication."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l156'
# Review existing portfolio contracts without upgrading their coverage.
portfolio=[]
for n,label,path,status in [(151,'Classification: rel-trial/study-outcome','labs/l151-reproduction.md','Existing complete fits; L156 fresh audit NOT_RUN'),(153,'Recommendation: rel-trial/study-sponsor','labs/l153-reproduction.md','Prior32-batch pilot; full fit INCOMPLETE; test NOT_RUN')]:
    source=P.parent/path;portfolio.append(dict(lesson=n,task=label,contract=path,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),coverage=status,availability='NOT_ESTABLISHED'))
(E/'portfolio-review.json').write_text(json.dumps(portfolio,indent=2))
paths=[E/name for name in ['preflight.json','train-dependencies.npz','val-dependencies.npz','test-dependencies.npz','train-labels.npz','val-labels.npz','test-labels.npz','label-events.npz','correction-protocol.json','protocol.json','source-manifest.json','portfolio-review.json']]
paths += [E/'fe/sql-audit.json']
paths += [E/f'{lane}/seed-{seed}/{name}' for lane in ['paper','fit_horizon'] for seed in range(5) for name in ['result.json','predictions.npz','completed.json','cost.json','temporal-audit.json','audit-l156.json','diagnostics.json','audit.json','checkpoint-audit.json']]
m=dict(scope='Ten primary full fits; notebook validation excluded; SHA256 integrity is not an independent human signature',files={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
f=E/'input-manifest.json'
if f.exists():assert json.loads(f.read_text())==m,'Frozen primary evidence changed'
else:f.write_text(json.dumps(m,indent=2)+'\n')
print('Pinned',len(paths),'primary inputs')

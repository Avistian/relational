"""Seal immutable input packet, upstream source and complete candidate schedule."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l192';Q=E/'packet';S=P/'sources/l192'
# Compact source for standalone primitive re-execution and readable full pipeline appendix.
for n in ['preprocessing.py','estimator.py','config.py','constants.py']:
 (Q/n).write_bytes((S/'rdblearn/rdblearn'/n).read_bytes())
protocol=json.loads((E/'protocol.json').read_text())
schedule=[dict(seed=s,depth=d,backend=b,status='NOT_RUN',validation_auc=None,test_auc=None) for s in protocol['seeds'] for d in protocol['depths'] for b in protocol['backends']]
(E/'candidate-schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
manifest=dict(files={p.relative_to(Q).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Q.rglob('*')) if p.is_file()})
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Authenticate original source and task extraction; environment setup outside wrappers charged conservatively.
budget=json.loads((E/'local-budget.json').read_text())
if not any(a.get('kind')=='early_inspection' for a in budget['attempts']):
 budget['attempts'].append(dict(kind='early_inspection',command=['initial dependency import probes'],seconds=5,status='ACCOUNTED_ALLOWANCE'))
 # The wrapper has an active reservation; do not mutate it from the child. Separate receipt instead.
 (E/'early-inspection-cost.json').write_text(json.dumps(dict(seconds=5,cloud_usd=0,scope='Conservative allowance for initial unwrapped imports'),indent=2)+'\n')
print('Packet files',len(manifest['files']),'planned candidates',len(schedule))

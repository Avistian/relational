"""Freeze the stopped experiment ledger and all offline audit inputs."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l193';Q=E/'packet';S=P/'sources/l193'
tasks=json.loads((Q/'tasks.json').read_text());protocol=json.loads((Q/'protocol.json').read_text())
probe=json.loads((Q/'preprocessing.json').read_text());assert probe['status']=='FAIL'
ledger=[dict(task=t['id'],seed=s,status='NOT_RUN',score=None,reason='Shared released preprocessing invariant failed before task dispatch',selected_candidate=None) for t in tasks for s in protocol['seeds']]
(Q/'runs.json').write_text(json.dumps(ledger,indent=2)+'\n')
for name,source in [('paper.html',S/'paper.html'),('preprocessing.py',S/'rdblearn/rdblearn/preprocessing.py'),('source-ledger.json',S/'source-ledger.json'),('validation-schedule.json',E/'validation-schedule.json'),('test-schedule.json',E/'test-schedule.json')]:
 (Q/name).write_bytes(source.read_bytes())
manifest=dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Q.iterdir()) if p.is_file()},scope='Immutable original-source diagnostic observations, paper reference and unrun full schedule')
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Sealed',len(manifest['files']),'packet files; 63 explicit unrun seed records')

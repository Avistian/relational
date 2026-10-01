"""Exercise actual reservation code without importing Modal or dispatching work."""
import ast,fcntl,json,tempfile,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
source=(R/'modal/l164_repro.py').read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reserve')
with tempfile.TemporaryDirectory(prefix='l164-budget-check-') as tmp:
 root=Path(tmp);(root/'labs/evidence/l164').mkdir(parents=True);(root/'worker.py').write_text('version1')
 p=root/'labs/evidence/l164/budget.json'
 b=dict(cap_usd=10,overhead_reserve_usd=.35,untouched_reserve_usd=2,reservations=[],source_hashes={'worker.py':hashlib.sha256(b'version1').hexdigest()})
 p.write_text(json.dumps(b));ns=dict(ROOT=root,RATE=.00034544,json=json,fcntl=fcntl,hashlib=hashlib)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-reserve>','exec'),ns)
 reserve=ns['reserve'];reserve('pilot',1500)
 for phase,seconds in [('pilot',1),('full',196000)]:
  try:reserve(phase,seconds)
  except AssertionError:pass
  else:raise AssertionError('Duplicate or excessive reservation passed')
 (root/'worker.py').write_text('changed')
 try:reserve('changed-worker',1)
 except AssertionError:pass
 else:raise AssertionError('Changed worker code was accepted')
 assert len(json.loads(p.read_text())['reservations'])==1
print('PASS: duplicate dispatch, aggregate cap, changed-worker rejection; no Modal imports')

import json,tempfile
from pathlib import Path
from _guard_l200 import reserve
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'budget.json'
 p.write_text(json.dumps(dict(cap_usd=10,planned_stop_usd=8,overhead_reserve_usd=2,max_worker_seconds=6000,source_hashes={},reservations=[])))
 assert abs(reserve(p,tmp,'pilot',600)-630*.00028372)<1e-12
 for phase,seconds in [('pilot',600),('large',5400),('zero',0),('float',1.5)]:
  try:reserve(p,tmp,phase,seconds)
  except ValueError:pass
  else:raise AssertionError('Unsafe reservation accepted')
 b=json.loads(p.read_text());b['source_hashes']={'changed.py':'bad'};p.write_text(json.dumps(b));(Path(tmp)/'changed.py').write_text('x')
 try:reserve(p,tmp,'changed',100)
 except ValueError:pass
 else:raise AssertionError('Changed source accepted')
print('PASS: duplicate, aggregate runtime, invalid timeout and source drift gates')

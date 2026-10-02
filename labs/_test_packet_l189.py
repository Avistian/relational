import json,tempfile,shutil
from pathlib import Path
from _audit_l189 import audit
from relkit.gaps_l189 import admission,priority,sensitivity
E=Path(__file__).resolve().parent/'evidence/l189';manifest=json.loads((E/'input-manifest.json').read_text())
r=audit(E/'packet',manifest,admission,priority,sensitivity)
assert r['case_count']==3 and r['weight_scenarios']==27 and r['status']=='COMPLETE_SELECTED_AUDIT'
assert r['ranking'][0]['id']=='temporal'
assert all(x['cost_gate']['status']=='NOT_ESTABLISHED' for x in r['ranking'])
for kind in ['missing','changed','extra']:
 with tempfile.TemporaryDirectory() as td:
  q=Path(td)/'packet';shutil.copytree(E/'packet',q)
  if kind=='missing':(q/'cases.json').unlink()
  elif kind=='changed':(q/'cases.json').write_text('[]')
  else:(q/'unlisted.json').write_text('{}')
  try:audit(q,manifest,admission,priority,sensitivity)
  except ValueError:pass
  else:raise AssertionError('Corrupt packet accepted: '+kind)
print('PASS packet contracts')

"""Ensure independent paper scoring rejects plausible corrupt evidence."""
import json,shutil,tempfile
from pathlib import Path
import numpy as np
from _audit_paper_b14 import audit
P=Path(__file__).resolve().parent;E=P/'evidence/b14';expected=json.loads((E/'official-keys.json').read_text());checks=[]
for mutation in ['missing-query','duplicate-key','wrong-label','wrong-score','test-selection']:
 with tempfile.TemporaryDirectory(prefix='b14-mutation-') as td:
  root=Path(td);shutil.copytree(E/'full',root,dirs_exist_ok=True)
  file=root/'r1-test.npz'
  with np.load(file) as z:data={k:z[k].copy() for k in z.files}
  if mutation=='missing-query':data={k:v[:-1] for k,v in data.items()}
  if mutation=='duplicate-key':data['entity'][0]=data['entity'][1];data['date'][0]=data['date'][1]
  if mutation=='wrong-label':data['label'][0]=1-data['label'][0]
  if mutation=='wrong-score':data['prediction']=1-data['prediction']
  if mutation=='test-selection':
   p=root/'result.json';r=json.loads(p.read_text());r['selected']=r['trials'][0]['config_id'];p.write_text(json.dumps(r))
  np.savez(file,**data)
  try:audit(root,expected)
  except (ValueError,AssertionError):checks.append(mutation)
  else:raise AssertionError('Corruption admitted: '+mutation)
r=dict(status='PASS',corruptions_rejected=checks);(E/'paper-corruption-tests.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

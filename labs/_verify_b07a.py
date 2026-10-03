"""Evidence corruption interventions and canonical source-gate tests."""
import hashlib,json,shutil,tempfile,subprocess,sys
from pathlib import Path
import numpy as np
from _audit_b07a import audit_course
from _source_b07a import audit_sources
P=Path(__file__).resolve().parent;E=P/'evidence/b07a'
base=audit_course(P,E/'runs');assert base==json.loads((E/'course-audit.json').read_text())
checks=[]
with tempfile.TemporaryDirectory(prefix='b07a-audit-') as td:
 root=Path(td);runs=root/'runs';shutil.copytree(E/'runs',runs);original=(runs/'results.json').read_text();report=json.loads(original)
 def rejected(label):
  try:audit_course(P,runs)
  except (AssertionError,ValueError,KeyError):checks.append(label)
  else:raise AssertionError('Accepted corruption: '+label)
 report['records'].pop();(runs/'results.json').write_text(json.dumps(report));rejected('missing arm');(runs/'results.json').write_text(original)
 report=json.loads(original);report['records'].append(report['records'][0]);(runs/'results.json').write_text(json.dumps(report));rejected('duplicate arm');(runs/'results.json').write_text(original)
 report=json.loads(original);record=report['records'][0];path=runs/record['file'];saved=path.read_bytes();v=dict(np.load(path));v['row_id']=v['row_id'][::-1];np.savez_compressed(path,**v);record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(runs/'results.json').write_text(json.dumps(report));rejected('changed row IDs with updated local checksum');path.write_bytes(saved);(runs/'results.json').write_text(original)
 report=json.loads(original);record=report['records'][0];v=dict(np.load(path));v['target']=1-v['target'];np.savez_compressed(path,**v);record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(runs/'results.json').write_text(json.dumps(report));rejected('changed targets with updated checksum');path.write_bytes(saved);(runs/'results.json').write_text(original)
 report=json.loads(original);record=report['records'][0];v=dict(np.load(path));v['probability'][0,0]=float('nan');np.savez_compressed(path,**v);record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(runs/'results.json').write_text(json.dumps(report));rejected('nonfinite probability with updated checksum');path.write_bytes(saved);(runs/'results.json').write_text(original)
 report=json.loads(original);report['records'][1]['predictor_sha256']='unpaired';(runs/'results.json').write_text(json.dumps(report));rejected('unpaired generated model');(runs/'results.json').write_text(original)
 source=root/'source';shutil.copytree(P/'sources/b07a',source);target=source/'hyperfast/hyperfast/model.py';target.write_bytes(target.read_bytes()+b'changed')
 try:audit_sources(source)
 except AssertionError:checks.append('changed source bytes')
 else:raise AssertionError('Accepted source corruption')
result=dict(status='PASS',audited_predictions=base['predictions'],rejected_interventions=checks,source_gate=audit_sources(P/'sources/b07a')['status'])
(P/'_verify_b07a_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

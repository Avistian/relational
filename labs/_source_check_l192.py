"""Confirm the loaded preprocessing and FastDFS bytes equal the frozen release."""
import hashlib,json,sys
from pathlib import Path
import fastdfs,rdblearn
import rdblearn.preprocessing,rdblearn.estimator,rdblearn.config
P=Path(__file__).resolve().parent;S=P/'sources/l192';checked={}
for package,base in [('fastdfs',Path(fastdfs.__file__).parent),('rdblearn',Path(rdblearn.__file__).parent)]:
 frozen=S/package/('rdblearn' if package=='rdblearn' else '')
 for source in sorted(frozen.rglob('*.py')):
  actual=base/source.relative_to(frozen)
  assert actual.read_bytes()==source.read_bytes(),str(actual)
  checked[package+'/'+str(source.relative_to(frozen))]=hashlib.sha256(actual.read_bytes()).hexdigest()
result=dict(status='PASS',loaded_source_files=len(checked),files=checked,scope='Loaded package code matches source ledger; not a backend or checkpoint audit')
(P/'evidence/l192/loaded-source-check.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],result['loaded_source_files'])

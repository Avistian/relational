"""Authenticate extracted sources against downloaded archives and their ledgers."""
import hashlib,json,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b14';count=0
for item in json.loads((S/'source-ledger.json').read_text()):assert hashlib.sha256((S/item['path']).read_bytes()).hexdigest()==item['sha256']
with zipfile.ZipFile(S/'relarena-v0.0.1.zip') as z:
 for n in z.namelist():
  if n.endswith('/'):continue
  dest=S/'relarena'/Path(*Path(n).parts[1:]);assert dest.read_bytes()==z.read(n),str(dest);count+=1
wheel=json.loads((S/'tabpfn-wheel.json').read_text());assert hashlib.sha256((S/'tabpfn-8.0.8.whl').read_bytes()).hexdigest()==wheel['digests']['sha256']
with zipfile.ZipFile(S/'tabpfn-8.0.8.whl') as z:
 for n in z.namelist():
  if n.endswith('.py') or 'LICENSE' in n:assert (S/'tabpfn'/n).read_bytes()==z.read(n);count+=1
original=S/'original-rdblearn';ledger=json.loads((original/'source-ledger.json').read_text())
for name,digest in ledger['files'].items():assert hashlib.sha256((original/name).read_bytes()).hexdigest()==digest;count+=1
r=dict(status='PASS',authenticated_source_files=count,original_rdblearn_commit=ledger['commit'],relarena_commit='e89002200e18be6d8d7a55f8a5ab50c993ce4d5d',tabpfn_wheel='8.0.8',historical_training_identity='NOT_ESTABLISHED');(P/'evidence/b14/source-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

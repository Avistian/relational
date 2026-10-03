"""Authenticate bundled released source and extract the exact official task archive.

All source bytes are distributed with this lesson. Only a missing task archive is
re-fetched; a changed source or archive fails instead of silently updating pins.
"""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l192';S=P/'sources/l192'
ledger=json.loads((S/'source-ledger.json').read_text())
if not (S/'task.zip').exists():
 with urllib.request.urlopen('https://relbench.stanford.edu/download/rel-trial/tasks/study-outcome.zip',timeout=45) as r:(S/'task.zip').write_bytes(r.read())
for name,digest in ledger['files'].items():
 path=P/name
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Source identity mismatch: '+name)
with zipfile.ZipFile(S/'task.zip') as z:
 for name in z.namelist():
  if name.endswith('.parquet'):
   path=E/'task'/Path(name).name;path.parent.mkdir(parents=True,exist_ok=True)
   data=z.read(name)
   if path.exists() and path.read_bytes()!=data:raise ValueError('Task extraction differs from archive')
   path.write_bytes(data)
print('Authenticated',len(ledger['files']),'source files and all task split bytes')

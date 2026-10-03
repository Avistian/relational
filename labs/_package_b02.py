"""Archive pinned executable source and authenticate retained raw run files."""
import hashlib,json,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b02';S=P/'sources/b02/upstream'
lock=json.loads((E/'source-lock.json').read_text())
entries={}
for name,digest in lock['source_files'].items():
 b=(S/name).read_bytes();assert hashlib.sha256(b).hexdigest()==digest;entries['labs/sources/b02/upstream/'+name]=b
for name,digest in lock['data_files'].items():
 b=(S/'data'/name).read_bytes();assert hashlib.sha256(b).hexdigest()==digest;entries['labs/sources/b02/upstream/data/'+name]=b
for name in ['labs/_worker_b02.py','modal/b02_repro.py','labs/_source_b02.py','labs/relkit/embeddings_b02.py','labs/b02-reproduction.md','labs/evidence/b02/source-lock.json','labs/evidence/b02/budget.json','labs/evidence/b02/admission.json']:
 entries[name]=(R/name).read_bytes()
with zipfile.ZipFile(E/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name,b in sorted(entries.items()):
  info=zipfile.ZipInfo(name,(2026,10,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
raw={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (E/'fresh').rglob('*') if p.is_file()}
(E/'raw-manifest.json').write_text(json.dumps(raw,indent=2)+'\n')
print('Pinned source archive',len(entries),'files; raw run manifest',len(raw),'files')

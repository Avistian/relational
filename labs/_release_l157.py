"""Build a deterministic portable release, excluding raw inputs/caches/weights."""
import hashlib,json,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'releases/l157-f1-audit'
def allowed(p):
 rel=p.relative_to(D)
 return p.is_file() and '__pycache__' not in rel.parts and p.suffix not in ['.pyc','.pt','.parquet','.zip'] and not str(rel).startswith('labs/sources/l129/f1/') and p.name!='release-manifest.json'
files=sorted(p for p in D.rglob('*') if allowed(p))
m={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(D/'release-manifest.json').write_text(json.dumps(m,indent=2)+'\n');files.append(D/'release-manifest.json')
with zipfile.ZipFile(P/'releases/l157-f1-audit.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(files):
  info=zipfile.ZipInfo(str(p.relative_to(D)),date_time=(2026,10,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes())
print('Packaged',len(files),'files; SHA256',hashlib.sha256((P/'releases/l157-f1-audit.zip').read_bytes()).hexdigest())

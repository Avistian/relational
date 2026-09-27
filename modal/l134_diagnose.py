"""Read-only archive diagnosis, bounded by USD3 recovery/overhead allocation."""
from pathlib import Path
import modal
app=modal.App('l134-archive-diagnosis');v=modal.Volume.from_name('l134-scale-evidence')
@app.function(timeout=180,cpu=.25,memory=512,retries=0,volumes={'/evidence':v})
def inspect():
 import hashlib,zipfile,json
 p=Path('/evidence/attempt-1/db.zip');h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 r=dict(bytes=p.stat().st_size,sha256=h.hexdigest(),zip=zipfile.is_zipfile(p))
 if r['zip']:
  with zipfile.ZipFile(p) as z:r['members']=[dict(name=x.filename,bytes=x.file_size) for x in z.infolist()]
 else:r['head']=p.read_bytes()[:200].decode(errors='replace')
 return r
@app.local_entrypoint()
def main():
 import json
 r=inspect.remote();print(r);Path('labs/_archive_probe_l134.json').write_text(json.dumps(r,indent=2))

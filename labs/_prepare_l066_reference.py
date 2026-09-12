"""Prepare two checksum-pinned reference dependencies without changing the runtime."""
import hashlib,io,json,urllib.request,zipfile
from pathlib import Path

def prepare(root):
 root=Path(root);folder=root/'data/cache/l066-reference-deps';folder.mkdir(parents=True,exist_ok=True)
 pins=json.loads((root/'_sources_l066_v2.json').read_text())['reference_dependencies']
 marker=folder/'installed.json'
 if not marker.exists() or json.loads(marker.read_text())!=pins:
  for pin in pins.values():
   data=urllib.request.urlopen(pin['url']).read()
   if hashlib.sha256(data).hexdigest()!=pin['sha256']:raise RuntimeError('Reference dependency checksum mismatch')
   zipfile.ZipFile(io.BytesIO(data)).extractall(folder)
  marker.write_text(json.dumps(pins,sort_keys=True))
 return folder

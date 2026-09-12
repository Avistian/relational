"""Pinned complete inference source provenance and download-only cache helper."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent
COMMIT='a6e75afb82d13e7abb46deade3669c4106b3d636'
def ensure_source():
 manifest=json.loads((ROOT/'_sources_l068_v2.json').read_text());base=ROOT/'data/cache/l068-official-source'
 for row in manifest['files']:
  p=base/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():urllib.request.urlretrieve(row['url'],p)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
 return base
if __name__=='__main__':print(ensure_source())

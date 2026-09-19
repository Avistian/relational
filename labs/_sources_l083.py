"""Restore missing reference source bytes from the immutable revision and verify their pinned hashes."""
import hashlib,json,urllib.request
from pathlib import Path
LAB=Path(__file__).resolve().parent
manifest=json.loads((LAB/'_sources_l083.json').read_text())
for name,digest in manifest['files'].items():
    path=LAB/'sources/l083'/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/williamleif/GraphSAGE/{manifest["revision"]}/{name}'
        body=urllib.request.urlopen(url).read()
        assert hashlib.sha256(body).hexdigest()==digest,name
        path.write_bytes(body)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
print('Pinned source files verified:',len(manifest['files']))

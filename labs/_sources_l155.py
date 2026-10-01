"""Verify the released FE bytes against pinned upstream URLs."""
import hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l129';m=json.loads((S/'manifest.json').read_text());checked=[]
for item in m['files']:
    if not item['url'].startswith('https://raw.githubusercontent.com/'):continue
    raw=urllib.request.urlopen(item['url'],timeout=30).read()
    assert hashlib.sha256(raw).hexdigest()==item['sha256']
    assert raw==(S/item['path']).read_bytes();checked.append(item)
r=dict(status='PASS',commit=m['commit'],files=checked,paper='https://arxiv.org/html/2407.20060v1#S6')
(P/'_sources_l155.json').write_text(json.dumps(r,indent=2));print('Verified FE sources',len(checked))

"""Verify all declared source bytes and retain the scope of each identity check."""
import hashlib,json
from pathlib import Path
P=Path(__file__).parent;R=P.parent
m=json.loads((P/'_sources_l134.json').read_text())
for name,digest in m['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((P/'sources/l134/paper.html').read_bytes()).hexdigest()==m['paper']['sha256']
scale=json.loads((P/'sources/l134/scale/manifest.json').read_text())
for name,item in scale.items():assert hashlib.sha256((P/'sources/l134/scale'/name).read_bytes()).hexdigest()==item['sha256']
b=json.loads((P/'_budget_l134.json').read_text())
for reservation in b['scale_reservations']:
 attempt=reservation['attempt'];folder=P/'sources/l134'/('failed-scale-'+attempt)
 path=folder/'_run_scale_l134.py' if folder.exists() else P/'_run_scale_l134.py'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==reservation['source_sha256'],attempt
r=dict(status='PASS',declared_files=len(m['files']),scale_upstream_files=len(scale),scale_attempt_source_hashes=len(b['scale_reservations']),historical_identity='NOT_ESTABLISHED')
(P/'_source_check_l134_results.json').write_text(json.dumps(r,indent=2));print(r)

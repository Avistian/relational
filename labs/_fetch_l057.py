"""Fetch only the pinned TabICL checkpoint; code installed separately from optional requirements."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def fetch(root=None):
    if root is None:root=ROOT/'data/cache/l057-checkpoint'
    s=json.loads((ROOT/'_sources_l057.json').read_text())['tabicl']
    p=Path(root)/s['filename'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():urllib.request.urlretrieve(s['url'],p)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
    return str(p.resolve())
if __name__=='__main__':print(fetch())

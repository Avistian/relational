"""Recheck core archived source bytes against their immutable upstream commit."""
from pathlib import Path
import urllib.request,hashlib,json
P=Path(__file__).resolve().parent
commit='61e9784ca76edeaa6e259ba0f836099608ff0586'
root='https://raw.githubusercontent.com/snap-stanford/ogb/'+commit+'/'
paths={'gnn.py':'examples/nodeproppred/arxiv/gnn.py','logger.py':'examples/nodeproppred/arxiv/logger.py','LICENSE':'LICENSE'}
verified={}
for name,remote in paths.items():
 live=urllib.request.urlopen(root+remote,timeout=30).read();local=(P/'sources/l116'/name).read_bytes();assert live==local,name;verified[name]=hashlib.sha256(local).hexdigest()
r={'status':'PASS','commit':commit,'fresh_byte_verification':verified};(P/'_upstream_l116_results.json').write_text(json.dumps(r,indent=2));print(r)

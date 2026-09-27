"""Pin the primary paper bytes and all inherited load-bearing source artifacts."""
import hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
out=P/'sources/l127';out.mkdir(parents=True,exist_ok=True)
url='https://arxiv.org/html/2407.20060v1'
raw=urllib.request.urlopen(url,timeout=45).read();(out/'paper.html').write_bytes(raw)
files=[P/'relkit/rdl_l117.py',P/'relkit/batch_audit_l123.py',P/'relkit/benchmark_l127.py',P/'_run_l117.py',P/'_run_l127.py',P/'requirements-l117-runtime.txt']+sorted((P/'sources/l117').rglob('*'))
sources={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
r=dict(paper=dict(url=url,sha256=hashlib.sha256(raw).hexdigest()),upstream_commit='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639',files=sources,identity='Pinned released implementation; exact paper training commit NOT_ESTABLISHED')
(P/'_sources_l127.json').write_text(json.dumps(r,indent=2));print('Pinned',len(sources),'files and primary paper HTML')

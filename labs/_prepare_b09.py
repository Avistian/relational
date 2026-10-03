"""Pin primary source bytes and obtain immutable source archives (not weights)."""
import requests,json,hashlib,tarfile,io,time
from pathlib import Path
P=Path(__file__).resolve().parent; S=P/'sources/b09'; start=time.monotonic()
ledger=[]
for name in ['tabfm','exaone','nori']:
 m=json.loads((S/f'{name}-commit.json').read_text());url=f'https://api.github.com/repos/{m["repo"]}/tarball/{m["commit"]}'
 r=requests.get(url,timeout=120);r.raise_for_status();(S/f'{name}.tar.gz').write_bytes(r.content)
 dest=Path('/tmp/b09-src')/name;dest.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
  for item in tar.getmembers():
   bits=item.name.split('/',1)
   if len(bits)==2 and bits[1]:item.name=bits[1];tar.extract(item,dest,filter='data')
 ledger.append(dict(name=name,url=url,sha256=hashlib.sha256(r.content).hexdigest(),commit=m['commit']))
for name,url in [('exaone-paper','https://arxiv.org/html/2608.25774v1'),('tabfm-paper','https://arxiv.org/html/2609.37959v1'),('nori-card','https://huggingface.co/Synthefy/Nori/raw/main/README.md'),('seldon','https://www.neuralk.ai/white-paper/seldon-foundation-made-tabular'),('nexus','https://fundamental.tech/nexus')]:
 try:
  r=requests.get(url,timeout=40);r.raise_for_status();(S/f'{name}.html').write_bytes(r.content)
  ledger.append(dict(name=name,url=url,sha256=hashlib.sha256(r.content).hexdigest()))
 except Exception as e:ledger.append(dict(name=name,url=url,status='UNAVAILABLE',error=str(e)))
(S/'provenance.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger,indent=2));print('preparation wall seconds',time.monotonic()-start)

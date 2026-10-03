"""Archive primary sources and authenticate all inherited raw database bytes."""
import hashlib,json,urllib.request,zipfile,io,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l181';S=P/'sources/l181'
rev='0d47fe0c8a1a51aaf97ab485f4a028e290f97c67'
url=f'https://codeload.github.com/stanford-star/relbench/zip/{rev}'
b=urllib.request.urlopen(url).read();z=zipfile.ZipFile(io.BytesIO(b));files={}
for n in z.namelist():
 rel='/'.join(n.split('/')[1:])
 if n.endswith('/') or not (rel.startswith('relbench/') or rel.startswith('examples/') or rel in ['pyproject.toml','README.md']):continue
 dest=S/'upstream'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n));files[str(dest.relative_to(R))]=hashlib.sha256(dest.read_bytes()).hexdigest()
urls={'relbench-v2.html':'https://arxiv.org/html/2602.12606v1','relgt-ac.html':'https://arxiv.org/html/2606.03040v1','pricing.html':'https://modal.com/pricing'}
for n,u in urls.items():
 d=urllib.request.urlopen(u).read();(S/n).write_bytes(d);files[str((S/n).relative_to(R))]=hashlib.sha256(d).hexdigest()
old=json.loads((P/'evidence/l171/input-manifest.json').read_text())['files'];raw={}
for src in sorted((P/'evidence/l171/db').glob('*.parquet')):
 key=str(src.relative_to(P));h=hashlib.sha256(src.read_bytes()).hexdigest();assert old[key]==h,key
 dest=E/'packet'/'db'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);raw[src.name]=h
(S/'source-ledger.json').write_text(json.dumps(dict(revision=rev,revision_meaning='publication-date snapshot; historical experiment commit NOT_ESTABLISHED',repository_url=url,archive_sha256=hashlib.sha256(b).hexdigest(),urls=urls,files=files,raw_sha256=raw,raw_provenance='L171 authenticated complete cached RelBench1.1.0 F1 archive; v2 identity to be audited'),indent=2)+'\n')
print('Archived',len(files),'sources and',len(raw),'authenticated tables')

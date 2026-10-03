"""Download the exact public checkpoint revision and authenticate against Hub LFS."""
import hashlib,json,requests,time
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b14';revision='24a16a89d245878b846555110985634aa2e656d7';name='tabpfn-v3-classifier-v3_default.ckpt';start=time.monotonic()
u=f'https://huggingface.co/api/models/Prior-Labs/tabpfn_3/tree/{revision}?recursive=true&expand=false';r=requests.get(u,timeout=30);r.raise_for_status();files=r.json();(S/'hub-files.json').write_text(json.dumps(files,indent=2))
item=next(x for x in files if x['path']==name);expected=item['lfs']['oid'];out=Path('/tmp/b14-weights');out.mkdir(exist_ok=True);file=out/name
if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
 with requests.get(f'https://huggingface.co/Prior-Labs/tabpfn_3/resolve/{revision}/{name}',stream=True,timeout=60) as response:
  response.raise_for_status()
  with file.open('wb') as f:
   for chunk in response.iter_content(1024*1024):f.write(chunk)
assert hashlib.sha256(file.read_bytes()).hexdigest()==expected
for f in ['LICENSE','README.md','config.json']:
 r=requests.get(f'https://huggingface.co/Prior-Labs/tabpfn_3/resolve/{revision}/{f}',timeout=30);r.raise_for_status();(S/('hub-'+f)).write_bytes(r.content)
record=dict(status='AUTHENTICATED',revision=revision,filename=name,sha256=expected,bytes=file.stat().st_size,seconds=time.monotonic()-start,historical_byte_identity='NOT_ESTABLISHED',note='Pinned available release; historical result CSV does not supply a checkpoint hash.')
(P/'evidence/b14/checkpoint.json').write_text(json.dumps(record,indent=2)+'\n');print(record)

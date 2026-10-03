import requests,json,hashlib,time
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b09';dest=Path('/tmp/b09-weights');start=time.monotonic();records=[]
for name,files in [('tabfm',['regression/config.json','regression/model.safetensors']),('exaone',['exaone-tabular-regressor-v1_default.safetensors']),('nori',['config.json','nori.pt'])]:
 meta=json.loads((S/f'{name}-hf.json').read_text())
 for f in files:
  path=dest/name/f;path.parent.mkdir(parents=True,exist_ok=True)
  url=f'https://huggingface.co/{meta["id"]}/resolve/{meta["sha"]}/{f}'
  if not path.exists():
   r=requests.get(url,stream=True,timeout=120);r.raise_for_status()
   partial=path.with_suffix(path.suffix+'.partial')
   with partial.open('wb') as o:
    for b in r.iter_content(1024*1024):o.write(b)
   partial.replace(path)
  h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();records.append(dict(model=name,file=f,repo=meta['id'],revision=meta['sha'],sha256=h,bytes=path.stat().st_size));print(name,f,path.stat().st_size,flush=True)
expected=S/'weights.json'
if expected.exists():
 old={(x['model'],x['file']):x['sha256'] for x in json.loads(expected.read_text())}
 assert all(old[(x['model'],x['file'])]==x['sha256'] for x in records),'Existing checkpoint digest changed'
else: expected.write_text(json.dumps(records,indent=2))
print('download seconds',time.monotonic()-start)

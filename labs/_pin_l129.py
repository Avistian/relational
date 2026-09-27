"""Freeze local implementations, runtime primitives and endpoint evidence."""
import hashlib,json,urllib.request,urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l129'
installed=Path('/tmp/l129-runtime/lib/python3.11/site-packages')
for name in ['dataset.py','mapper.py','stats.py']:
 (S/'frame'/name).write_bytes((installed/'torch_frame/data'/name).read_bytes())
(S/'historical_dataset.py').write_bytes((installed/'relbench/data/dataset.py').read_bytes())
probe=[]
for url in ['https://relbench.stanford.edu/staging_data/rel-f1/db.zip']:
 try:
  with urllib.request.urlopen(url,timeout=30) as r:probe.append(dict(url=url,status=r.status,final_url=r.url))
 except urllib.error.HTTPError as e:probe.append(dict(url=url,status=e.code,final_url=e.url))
(S/'historical-endpoint.json').write_text(json.dumps(probe,indent=2))
manifest=json.loads((S/'manifest.json').read_text());existing={r['path'] for r in manifest['files']}
for f in sorted((S/'frame').glob('*')):
 rel=f.relative_to(S).as_posix()
 if rel in existing:continue
 folder='torch_frame/data' if f.name in ['dataset.py','mapper.py','stats.py'] else 'torch_frame/gbdt'
 url='https://raw.githubusercontent.com/pyg-team/pytorch-frame/56f687ddf4bf1c4a7d7b72ab0ef4117493256199/'+('LICENSE' if f.name=='LICENSE' else f'{folder}/{f.name}')
 manifest['files'].append(dict(path=rel,url=url,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
manifest['frame_commit']='56f687ddf4bf1c4a7d7b72ab0ef4117493256199';manifest['licenses']={'frame':'MIT; included','user_study':'No license file in pinned repository tree; attributed original public release'}
(S/'manifest.json').write_text(json.dumps(manifest,indent=2))
files=[p for p in S.rglob('*') if p.is_file() and '__pycache__' not in str(p)]+[P/'relkit/manual_fe_l129.py',P/'relkit/fe_experiment_l129.py',P/'_run_l129.py',P/'requirements-l129-runtime.txt']
r=dict(status='PINNED',files={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},paper='https://arxiv.org/html/2407.20060v1',historical_endpoint=probe,exact_paper_identity='NOT_ESTABLISHED')
(P/'_sources_l129.json').write_text(json.dumps(r,indent=2));print('Pinned',len(files),'files',probe)

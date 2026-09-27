"""Freeze current leaderboard sources and published regression prediction archives."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l136';S.mkdir(parents=True,exist_ok=True)
COMMIT='584a03d518b2b655580ea8e1cfbbb26bec0a2841'
HF='d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600'
base=f'https://raw.githubusercontent.com/stanford-star/relbench/{COMMIT}/'
urls={
 'leaderboard.json':base+'leaderboard/leaderboard.json','submit.py':base+'relbench/submit.py','metrics.py':base+'relbench/metrics.py',
 'task_entity.py':base+'relbench/base/task_entity.py','LICENSE':base+'LICENSE',
 'regression_stds.json':f'https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/{HF}/regression_stds.json',
 'kapso.zip':'https://github.com/user-attachments/files/31459805/regression-kapso.zip',
 'gnn.zip':'https://github.com/user-attachments/files/31398757/gnn-regression.zip',
 'plurel.zip':'https://github.com/user-attachments/files/31533926/rt-plurel-regression.zip'}
for issue in [393,380,397]:urls[f'entry-{issue}.json']=base+f'leaderboard/entries/{issue}.json';urls[f'issue-{issue}.json']=f'https://api.github.com/repos/stanford-star/relbench/issues/{issue}'
kapso='30ddb51cb971a105f37d4089c447af59e3d1e861'
for name in ['README.md','EVALUATION_PROTOCOL.md']:urls['kapso-'+name]=f'https://raw.githubusercontent.com/Leeroo-AI/kapso/{kapso}/benchmarks/relbench/{name}'
existing=json.loads((P/'_sources_l136.json').read_text()) if (P/'_sources_l136.json').exists() else None
def fetch(item):
 name,url=item;path=S/name
 raw=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=90).read()
 if existing:assert hashlib.sha256(raw).hexdigest()==existing['files'][name]['sha256'],name
 path.write_bytes(raw)
 return name,dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:files=dict(ex.map(fetch,urls.items()))
manifest=dict(relbench_commit=COMMIT,hf_revision=HF,kapso_commit=kapso,files=files)
(P/'_sources_l136.json').write_text(json.dumps(manifest,indent=2));print('Archived',len(files),'sources; bytes',sum(x['bytes'] for x in files.values()))

"""Download all 38 historical model tables and three released ranking records."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent/'sources/b19a';m=json.loads((P/'manifest.json').read_text());sha=m['output_commit'];tree=json.loads((P/'output-tree.json').read_text())['tree']
paths=[x['path'] for x in tree if x['path'].endswith('.parquet') or x['path'] in [f'figures/leaderboard/cd_data_{metric}.json' for metric in ['crps','r2','crls']]]
def fetch(path):
 dest=P/'output'/path;dest.parent.mkdir(parents=True,exist_ok=True)
 raw=f'https://raw.githubusercontent.com/jonaslandsgesell/ScoringBenchOutput/{sha}/{path}'
 def get(url):
  with urllib.request.urlopen(url,timeout=60) as r:return r.read()
 pointer=get(raw);url=raw
 if pointer.startswith(b'version https://git-lfs'):
  fields=dict(line.split(' ',1) for line in pointer.decode().strip().splitlines());url=f'https://media.githubusercontent.com/media/jonaslandsgesell/ScoringBenchOutput/{sha}/{path}'
  data=dest.read_bytes() if dest.exists() else get(url)
  assert len(data)==int(fields['size']) and hashlib.sha256(data).hexdigest()==fields['oid'].split(':')[1]
 else:data=pointer
 dest.write_bytes(data)
 return dict(file='output/'+path,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:entries=list(pool.map(fetch,paths))
m['files']=[x for x in m['files'] if not x['file'].startswith('output/')]+entries;(P/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(len(entries),sum(x['bytes'] for x in entries))

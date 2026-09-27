"""Pin primary scale source (released task SQL, truncation, temporal sampler)."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).parent;D=P/'sources/l134/scale';D.mkdir(parents=True,exist_ok=True)
base='https://raw.githubusercontent.com/snap-stanford/relbench/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/'
urls={name:base+remote for name,remote in {'dataset.py':'relbench/base/dataset.py','task_stack.py':'relbench/tasks/stack.py','dataset_stack.py':'relbench/datasets/stack.py','db_hashes.json':'relbench/datasets/hashes.json','task_hashes.json':'relbench/tasks/hashes.json'}.items()}
urls['neighbor_sampler.py']='https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.6.1/torch_geometric/sampler/neighbor_sampler.py'
urls['sampler_utils.py']='https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.6.1/torch_geometric/sampler/utils.py'
def fetch(pair):
 name,url=pair;raw=urllib.request.urlopen(url,timeout=40).read();(D/name).write_bytes(raw);return name,dict(url=url,sha256=hashlib.sha256(raw).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:r=dict(pool.map(fetch,urls.items()))
(D/'manifest.json').write_text(json.dumps(r,indent=2));print('Pinned',len(r),'source files')

"""Download immutable upstream sources; record primary-source and archive provenance."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'sources/l132';D.mkdir(parents=True,exist_ok=True)
COMMIT='9aa346267c2e1c560bd92da07d6f4ad1ca2f0639'
base=f'https://raw.githubusercontent.com/stanford-star/relbench/{COMMIT}/'
urls={f:base+'examples/'+f for f in ['gnn_link.py','idgnn_link.py','model.py','text_embedder.py']}
urls.update({f:base+'relbench/modeling/'+f for f in ['graph.py','loader.py','nn.py','utils.py']})
urls.update({'trial_task.py':base+'relbench/tasks/trial.py','dataset_registry.json':base+'relbench/datasets/dataset_registry.json','task_registry.json':base+'relbench/tasks/task_registry.json','LICENSE':base+'LICENSE','paper.html':'https://arxiv.org/html/2407.20060v1','idgnn.html':'https://snap.stanford.edu/idgnn/'})
def fetch(item):
 name,url=item
 try:
  data=urllib.request.urlopen(url,timeout=60).read();(D/name).write_bytes(data)
  return name,dict(url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
 except Exception as e:return name,dict(url=url,error=str(e))
r=dict(concurrent.futures.ThreadPoolExecutor(6).map(fetch,urls.items()))
(D/'manifest.json').write_text(json.dumps(dict(commit=COMMIT,sources=r),indent=2));print(json.dumps(r,indent=2))

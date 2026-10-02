"""Pin the original F1 label oracle and all primary-source bytes."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;E=P/'evidence/l175';S=P/'sources/l175'
r=pd.read_parquet(Path.home()/'.cache/relbench/rel-f1/db/results.parquet')
d=pd.read_parquet(Path.home()/'.cache/relbench/rel-f1/db/drivers.parquet')
np.savez_compressed(E/'label-oracle.npz',driver=r.driverId.to_numpy(),time=(r.date.astype('datetime64[ns]').astype('int64')//10**9).to_numpy(),status=r.statusId.to_numpy(),driver_ids=d.driverId.to_numpy())
urls={'paper.html':'https://arxiv.org/html/2510.06377v1','model-card.md':'https://huggingface.co/stanford-star/rt-v1/blob/299701dedae451f3dfa40717b831d9dc17c0e4e7/README.md','pricing.html':'https://modal.com/pricing','relbench-f1-task.py':'https://github.com/stanford-star/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/tasks/f1.py'}
items=[]
for p in sorted(S.rglob('*')):
 if not p.is_file() or p.name=='source-ledger.json':continue
 n=p.relative_to(S).as_posix()
 url=urls.get(n,'https://github.com/stanford-star/relational-transformer/blob/8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb/'+n.removeprefix('upstream/'))
 if n.startswith('example_'):url=url.replace('/example_','/scripts/example_')
 if n in ['column_index.json','table_info.json']:url='https://huggingface.co/datasets/hvag976/relational-transformer/blob/e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee/rel-f1/'+n
 items.append(dict(file=n,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),url=url))
(S/'source-ledger.json').write_text(json.dumps(dict(retrieved='2026-10-02',code_revision='8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb',data_revision='e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee',checkpoint_revision='299701dedae451f3dfa40717b831d9dc17c0e4e7',sources=items),indent=2)+'\n')
(E/'raw-oracle-manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path.home()/'.cache/relbench/rel-f1/db/results.parquet',Path.home()/'.cache/relbench/rel-f1/db/drivers.parquet']},indent=2)+'\n')
print('Pinned',len(items),'source files;',len(r),'raw results for independent label reconstruction')

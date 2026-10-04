"""Bounded read-only fetch of declared original-suite result artifacts, no fits."""
import json,hashlib,requests
from pathlib import Path
P=Path(__file__).resolve().parent/'sources/b19'
methods=['LinearModel','ExtraTrees','LightGBM','TA-RealMLP','TA-TabPFN-2.6','TA-TabICLv2']
receipts=[]
for method in methods:
 for part in ['metadata.yaml','results/model_results.parquet','results/hpo_results.parquet']:
  url=f'https://data.tabarena.ai/cache/artifacts/beyond_iid_benchmark_2026/methods/{method}/{part}'
  name=method+'--'+part.replace('/','-');target=P/name
  if target.exists():data=target.read_bytes();status=200
  else:
   r=requests.get(url,timeout=30);status=r.status_code;data=r.content
   if status==200 and len(data)<40_000_000:target.write_bytes(data)
  receipts.append(dict(method=method,part=part,url=url,status=status,bytes=len(data),file=name if target.exists() else None,sha256=hashlib.sha256(data).hexdigest()))
(P/'result-fetch.json').write_text(json.dumps(receipts,indent=2)+'\n')
print([(r['method'],r['part'],r['status'],r['bytes']) for r in receipts])

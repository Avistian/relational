"""Collect isolated full notebook evidence; retain executed notebook locally."""
import json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;v=modal.Volume.from_name('l156-regression-evidence')
def collect(name):
 dest=P/('results' if name.endswith(('.ipynb','.pt')) else 'evidence')/'l156/notebook'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix(dest.suffix+'.partial')
 with tmp.open('wb') as f:
  for chunk in v.read_file('notebook/'+name):f.write(chunk)
 tmp.replace(dest)
names=['execution.json','cost.json','executed.ipynb','l156-full-report.json']
names += [f'l156-full/{lane}/seed-{seed}/{name}' for lane in ['released','fit_horizon'] for seed in range(5) for name in ['result.json','predictions.npz','audit-l156.json','diagnostics.json','temporal-audit.json','checkpoint-audit.json','audit.json','selected.pt']]
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(collect,names))
print(json.loads((P/'evidence/l156/notebook/execution.json').read_text()))

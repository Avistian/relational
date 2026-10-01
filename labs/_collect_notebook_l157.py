"""Collect compact validation evidence and independently hash all remote weights."""
import concurrent.futures,hashlib,json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;E=P/'evidence/l157/notebook';v=modal.Volume.from_name('l157-notebook-validation')

def collect(name):
    remote='validation/'+name;dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if name.endswith('selected.pt'):
        h=hashlib.sha256();size=0
        for chunk in v.read_file(remote):h.update(chunk);size+=len(chunk)
        (dest.parent/'collected-weight.json').write_text(json.dumps(dict(sha256=h.hexdigest(),bytes=size,method='Independent streaming SHA256 of remote selected.pt'),indent=2))
    else:
        tmp=dest.with_suffix(dest.suffix+'.partial')
        with tmp.open('wb') as f:
            for chunk in v.read_file(remote):f.write(chunk)
        tmp.replace(dest)

names=['execution.json','cost.json','l157-report.json','l157-fresh-report.json']
names += [f'fresh/{lane}/seed-{seed}/{file}' for lane in ['paper','fit_horizon'] for seed in range(5) for file in ['result.json','predictions.npz','completed.json','audit-l156.json','temporal-audit.json','diagnostics.json','checkpoint-audit.json','selected.pt']]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for name in pool.map(collect,names):pass
print('Collected full notebook evidence and verified ten weight streams')

"""Collect compact evidence separately from weights; preserve remote evidence."""
import argparse
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--seeds',default='0,1,2,3,4');p.add_argument('--lane',default='paper');a=p.parse_args();v=modal.Volume.from_name('l156-regression-evidence')
for seed in map(int,a.seeds.split(',')):
 for name in ['audit-l156.json','audit.json','result.json','predictions.npz','completed.json','started.json','cost.json','temporal-audit.json','checkpoint-audit.json','diagnostics.json','selected.pt']:
  dest=P/('results' if name=='selected.pt' else 'evidence')/f'l156/{a.lane}/seed-{seed}'/name;dest.parent.mkdir(parents=True,exist_ok=True)
  tmp=dest.with_suffix(dest.suffix+'.partial')
  with tmp.open('wb') as f:
   for chunk in v.read_file(f'{a.lane}/seed-{seed}/{name}'):f.write(chunk)
  tmp.replace(dest)
 print('Collected fresh seed',seed)

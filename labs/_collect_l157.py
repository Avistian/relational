import argparse,shutil
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;D=P/'releases/l157-f1-audit';p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();v=modal.Volume.from_name('l157-f1-contribution')
for lane in (['paper'] if a.pilot else ['paper','fit_horizon']):
 for seed in (range(1) if a.pilot else range(5)):
  for name in ['audit-l156.json','audit.json','result.json','predictions.npz','completed.json','started.json','cost.json','temporal-audit.json','checkpoint-audit.json','diagnostics.json','selected.pt']:
   dest=(P/'results/l157' if name=='selected.pt' else D/'labs/evidence/l157')/lane/f'seed-{seed}'/name;dest.parent.mkdir(parents=True,exist_ok=True)
   temp=dest.with_suffix(dest.suffix+'.partial')
   with temp.open('wb') as f:
    for chunk in v.read_file(f'{lane}/seed-{seed}/{name}'):f.write(chunk)
   temp.replace(dest)
  print('Collected',lane,seed)

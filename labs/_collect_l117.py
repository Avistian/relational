"""Collect compact evidence and selected weights without mixing the two."""
import argparse
from pathlib import Path
import modal
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','paper'],default='paper');a=p.parse_args()
v=modal.Volume.from_name('l117-rdl-evidence')
for seed in ([100] if a.mode=='pilot' else range(5)):
    for name in ['audit.json','result.json','predictions.npz','completed.json','selected.pt']:
        dest=P/('results' if name=='selected.pt' else 'evidence')/'l117'/a.mode/f'seed-{seed}'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('wb') as f:
            for chunk in v.read_file(f'{a.mode}/seed-{seed}/{name}'):f.write(chunk)
    print('Collected',a.mode,seed)

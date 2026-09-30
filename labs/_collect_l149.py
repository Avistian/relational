"""Collect all artifacts from a completed declared phase."""
import argparse,json
from pathlib import Path
import modal
P=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','search','final'],required=True);a=p.parse_args()
v=modal.Volume.from_name('l149-error-analysis-evidence')
assert a.mode in ['pilot','final']
pairs=[('lr005-full',999)] if a.mode=='pilot' else [('lr005-full',s) for s in range(5)]
for c,s in pairs:
 for name in ['audit.json','result.json','predictions.npz','completed.json','selected.pt']:
  dest=P/('results' if name=='selected.pt' else 'evidence')/'l149'/a.mode/c/f'seed-{s}'/name
  dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.with_suffix(dest.suffix+'.partial').open('wb') as fp:
   for chunk in v.read_file(f'{a.mode}/{c}/seed-{s}/{name}'):fp.write(chunk)
  dest.with_suffix(dest.suffix+'.partial').replace(dest)
 print('Collected',a.mode,c,s)

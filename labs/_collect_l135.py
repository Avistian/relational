"""Collect all artifacts from a completed declared phase."""
import argparse,json
from pathlib import Path
import modal
P=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','search','final'],required=True);a=p.parse_args()
v=modal.Volume.from_name('l135-tuning-evidence')
protocol=json.loads((P/'_protocol_l135.json').read_text())
if a.mode=='pilot':pairs=[(protocol['default'],999)]
elif a.mode=='search':pairs=[(c['id'],s) for c in protocol['configurations'] for s in protocol['search_seeds']]
else:
 f=json.loads((P/'evidence/l135/frozen.json').read_text());pairs=[(c,s) for c in dict.fromkeys([protocol['default'],f['winner']]) for s in protocol['final_seeds']]
for c,s in pairs:
 for name in ['audit.json','result.json','predictions.npz','completed.json','selected.pt']:
  dest=P/('results' if name=='selected.pt' else 'evidence')/'l135'/a.mode/c/f'seed-{s}'/name
  dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.with_suffix(dest.suffix+'.partial').open('wb') as fp:
   for chunk in v.read_file(f'{a.mode}/{c}/seed-{s}/{name}'):fp.write(chunk)
  dest.with_suffix(dest.suffix+'.partial').replace(dest)
 print('Collected',a.mode,c,s)

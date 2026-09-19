import argparse,json
from pathlib import Path
from relkit.contrastive_l072 import run_experiment

def config(preset):
    if preset=='paper':raise ValueError('No paper-fidelity preset: see l072-reproduction.md')
    return dict(seeds=(0,),datasets=('wine',),epochs=2) if preset=='smoke' else dict(seeds=tuple(range(5)),epochs=200)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer','paper'],default='smoke');p.add_argument('--device',default='cpu');a=p.parse_args()
    r=run_experiment(**config(a.preset),device=a.device)
    dest=Path(f'l072-{a.preset}-results.json');dest.write_text(json.dumps(r,indent=2));print(dest)

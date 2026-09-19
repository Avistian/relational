"""Local CPU sensitivity operator, same visible implementation as the lab."""
import argparse,json,os
from pathlib import Path
from relkit.carte_l074 import run_transfer
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer'],default='smoke');a=p.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    r=run_transfer(seeds=(0,),epochs=2) if a.preset=='smoke' else run_transfer(train_size=128,epochs=100)
    Path(f'_l074_{a.preset}_results.json').write_text(json.dumps(r,indent=2))
    print({'verified_here':a.preset+' local transfer','paper_benchmark':'INCOMPARABLE','fresh_YAGO_pretraining':'NOT_RUN'})

"""Run a declared bounded local panel; no cloud job or historical result rewrites."""
import argparse,json
from pathlib import Path
from relkit.openenv_l069_v2 import PRESETS69,run_experiment
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS69,default='closer');p.add_argument('--output',default=str(ROOT/'_verify_l069_v2_results.json'));a=p.parse_args()
 destination=Path(a.output)
 if destination.exists():raise FileExistsError('Choose a fresh --output path; measured artifacts are immutable')
 result=run_experiment(ROOT,PRESETS69[a.preset]);destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('WROTE',a.output,result['seconds'])

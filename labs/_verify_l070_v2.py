"""Fresh L070 corrected fits; preserves all historical inputs, no paid/cloud work."""
import argparse,json
from pathlib import Path
from _prepare_l070_v2 import prepare_inputs
from relkit.checkpoint_l070_v2 import run_experiment,PRESETS70
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS70,default='lab');p.add_argument('--output',default=str(Path(__file__).parent/'_verify_l070_v2_results.json'));a=p.parse_args()
 prepare_inputs(Path(__file__).parent)
 run_experiment(Path(__file__).parent,PRESETS70[a.preset],output=a.output)

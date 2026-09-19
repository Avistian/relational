"""Train the complete declared exam, or a distinct execution smoke."""
import argparse
from pathlib import Path
from relkit.exit_l080 import run_experiment,PRESETS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,default='exam');p.add_argument('--output',required=True);a=p.parse_args()
    run_experiment(Path(__file__).resolve().parent,a.preset,a.output)

"""Fresh L060 v2 evidence; refuses to overwrite an existing output."""
import argparse
from relkit.checkpoint_l060_v2 import run_checkpoint
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','lab','closer'],default='lab');p.add_argument('--output');a=p.parse_args();run_checkpoint(a.preset,a.output)

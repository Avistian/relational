"""Fresh historical Table7 reproduction using unchanged visible model/trainer."""
from _run_l135 import run as run_pinned
CONFIG=dict(id='lr005-full',lr=.005,fanout=[128,64])
def run(seed,epochs,output):
    return run_pinned(seed,epochs,output,CONFIG,evaluate_test=epochs==10)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=10);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.seed,a.epochs,a.output)

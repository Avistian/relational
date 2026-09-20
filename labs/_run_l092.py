"""Run the complete released ACM schedule and named Table3 KNN evaluation."""
import argparse,json,time
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent / "relkit"))
from han_l092 import load_acm,train_acm,summarize
p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0]);p.add_argument('--epochs',type=int,default=200);p.add_argument('--mode',choices=['release_node','paper_global'],default='release_node');p.add_argument('--output',default='labs/_paper_l092_results.json');args=p.parse_args()
root=Path(__file__).resolve().parent;manifest=json.loads((root/'_sources_l092.json').read_text());torch.set_num_threads(2)
data=load_acm(root/'data/l092',manifest);runs=[]
for seed in args.seeds:
 run,model=train_acm(data,seed,args.epochs,args.mode,True);runs.append(run)
 out=summarize(runs,data[-1]);out['requested_seeds']=args.seeds;out['command_mode']=args.mode
 if len(runs)<len(args.seeds):out['status']='PARTIAL_REQUESTED_SEEDS'
 Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
 print([(r['fraction'],r['macro_mean'],r['micro_mean']) for r in out['table3_acm']],flush=True)

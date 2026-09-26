"""Fresh local/Colab replay using the complete visible model. Preparation is separate."""
import argparse,json
from pathlib import Path
import torch
from relkit.scaling_l113 import train_run
p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer','paper'],default='smoke');p.add_argument('--root',default='labs/data/l113');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=0);p.add_argument('--max-seconds',type=int,default=2350);a=p.parse_args()
torch.set_num_threads(4);root=Path(a.root);blob=torch.load(root/'data.pt',weights_only=False);clusters=torch.load(root/'clusters.pt',weights_only=False);adj=torch.load(root/'adj.pt',weights_only=False)
seeds=range(10) if a.preset=='paper' else [a.seed];epochs={'smoke':2,'closer':50,'paper':50}[a.preset]
records=[train_run(blob['data'],clusters,adj,blob['split'],Path(a.output)/f'seed-{s}',s,epochs,a.device,a.max_seconds) for s in seeds]
print(json.dumps({'status':'COMPLETE' if all(r['status']=='COMPLETE' for r in records) else 'INCOMPLETE','runs':len(records),'epochs':epochs}))

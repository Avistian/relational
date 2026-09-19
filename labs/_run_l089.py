"""Fresh Cluster-GCN run. --preset paper is the full PPI Table 10 release recipe."""
import argparse,hashlib,importlib.metadata as md,json,platform,sys,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from cluster_gcn_l089 import load_ppi,preset,train
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer','paper'],default='smoke');p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);p.add_argument('--data',default='labs/data/l089');a=p.parse_args()
 out=Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit('Use a fresh output directory; no unchecked resume')
 out.mkdir(parents=True,exist_ok=True)
 cfg=preset(a.preset)
 meta={'preset':a.preset,'config':cfg,'seed':a.seed,'device':a.device,'python':sys.version,'machine':platform.machine(),'versions':{n:md.version(n) for n in ['torch','numpy','scipy','scikit-learn','pymetis']},'source_sha256':hashlib.sha256((Path(__file__).parent/'relkit/cluster_gcn_l089.py').read_bytes()).hexdigest(),'historical_parity':'INCOMPARABLE','paper_target':.9936}
 if a.device.startswith('cuda'):meta['gpu']=torch.cuda.get_device_name()
 (out/'run.json').write_text(json.dumps(meta,indent=2))
 d=load_ppi(a.data);r=train(d,cfg,a.seed,a.device,out)
 r['evidence_lane']='release_reconstruction' if a.preset=='paper' else 'teaching'
 r['historical_parity']='INCOMPARABLE'
 (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='curves'},indent=2))
if __name__=='__main__':main()

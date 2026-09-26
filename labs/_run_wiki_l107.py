import argparse,sys,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
import torch,numpy as np
from auth_l107 import authenticate_wiki_cache
from wiki_snapshot_l107 import wiki_data,train_snapshot,train_tgn_course
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['tgn','3600','86400'],default='3600');p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu');p.add_argument('--pilot',action='store_true');p.add_argument('--output',default=str(P/'evidence/l107/wiki'));p.add_argument('--data',default=str(P/'data/l102'));a=p.parse_args();torch.set_num_threads(1)
 authenticate_wiki_cache(a.data);nodes,edges,events,audit=wiki_data(a.data,a.seed);authenticate_wiki_cache(a.data);out=Path(a.output)/a.arm/f'seed-{a.seed}';out.mkdir(parents=True,exist_ok=True)
 identity={n:hashlib.sha256((P/'relkit'/n).read_bytes()).hexdigest() for n in ['snapshot_l107.py','wiki_snapshot_l107.py','tgn_l102.py']};identity.update({'audit':audit,'seed':a.seed,'arm':a.arm,'pilot':a.pilot,'torch':torch.__version__,'numpy':np.__version__,'device':a.device,'candidate_sha256':hashlib.sha256(events['negative'].tobytes()).hexdigest()})
 if (out/'identity.json').exists():raise RuntimeError('Output already exists; use a fresh directory to preserve previous evidence')
 if (out/'result.json').exists():raise RuntimeError('Existing run; choose a fresh output directory')
 (out/'identity.json').write_text(json.dumps(identity,indent=2));np.savez_compressed(out/'questions.npz',**events)
 if a.arm=='tgn':train_tgn_course(nodes,edges,events,a.seed,out,a.device,1 if a.pilot else 10,a.pilot)
 else:train_snapshot(nodes,edges,events,int(a.arm),a.seed,out,a.device,1 if a.pilot else 10,a.pilot)

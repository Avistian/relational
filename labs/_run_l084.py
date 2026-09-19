"""Fresh full Cora release-protocol runs. No implicit resume or cached score reuse."""
import argparse,hashlib,json,platform,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch,numpy,scipy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from gat_l084 import load_cora,train_cora,summarize
LAB=Path(__file__).resolve().parent

def worker(args):
 seed,max_epochs,out=args
 manifest=json.loads((LAB/'_sources_l084.json').read_text());data=load_cora(LAB,manifest)
 t=time.monotonic();r,model=train_cora(data,seed,max_epochs=max_epochs,return_model=True)
 r['seconds']=time.monotonic()-t
 if seed==0:
  torch.save(model.state_dict(),LAB/'l084-seed0.pt')
  with torch.no_grad():
   _,a=model.hidden[0](data[0],data[1],True)
  receiver=0;mask=data[1][0]==receiver
  (LAB/'_attention_l084.json').write_text(json.dumps({'seed':0,'head':0,'receiver':receiver,'senders':data[1][1,mask].tolist(),'coefficients':a[mask].tolist(),'meaning':'Evaluation weights; descriptive, not causal importance'},indent=2)+'\n')
 Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/f'seed-{seed:03d}.json').write_text(json.dumps(r)+'\n')
 return r

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--workers',type=int,default=4);p.add_argument('--max-epochs',type=int,default=100000);p.add_argument('--output',type=Path,default=LAB/'_paper_l084_results.json');a=p.parse_args()
 # Populate and verify the cache once before concurrent workers start.
 torch.set_num_threads(1)
 load_cora(LAB,json.loads((LAB/'_sources_l084.json').read_text()))
 tasks=[(i,a.max_epochs,LAB/'runs/l084') for i in range(a.seeds)];runs=[]
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(worker,tasks):
   runs.append(r);print(f"seed {r['seed']}: {r['test_accuracy']:.4f}, {r['epochs']} epochs, {r['seconds']:.1f}s",flush=True)
 result=summarize(runs);result.update(implementation_sha256=hashlib.sha256((LAB/'relkit/gat_l084.py').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((LAB/'_sources_l084.json').read_bytes()).hexdigest(),environment={'python':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__},max_epochs=a.max_epochs)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['mean'],result['sample_sd'])

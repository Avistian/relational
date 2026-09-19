"""Full 8-config x 10-fold MUTAG experiment, with pinned completed-run reuse."""
import argparse,hashlib,json,os,time,importlib.metadata as md
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from relkit.gin_l088 import load_mutag,train_fold,paper_grid,summarize_grid
LAB=Path(__file__).resolve().parent

def worker(job):
 config,fold,out,fingerprint=job;out=Path(out)
 name=f"h{config['hidden']}-b{config['batch_size']}-d{config['dropout']}-f{fold}"
 path=out/(name+'.json');npz=out/(name+'.npz')
 if path.exists():
  r=json.loads(path.read_text())
  if r['fingerprint']!=fingerprint or hashlib.sha256(npz.read_bytes()).hexdigest()!=r['logits_sha256']:
   raise ValueError('Existing evidence fingerprint mismatch: '+name)
  return name
 t=time.time();graphs=load_mutag(LAB/'data/l088');r,logits=train_fold(graphs,fold=fold,**config)
 np.savez_compressed(npz,logits=logits,labels=[graphs[i]['y'] for i in r['val_ids']],val_ids=r['val_ids'])
 r.update(fingerprint=fingerprint,seconds=time.time()-t,logits_sha256=hashlib.sha256(npz.read_bytes()).hexdigest())
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(path)
 return name

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--output',type=Path,default=LAB/'results/l088/paper');args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=True)
 identity={'source_sha256':hashlib.sha256((LAB/'relkit/gin_l088.py').read_bytes()).hexdigest(),
 'runtime':{x:md.version(x) for x in ['torch','numpy','scikit-learn']},'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
 jobs=[(c,f,str(args.output),fingerprint) for c in paper_grid() for f in range(10)]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(worker,j) for j in jobs]
  for i,f in enumerate(as_completed(futures)):print(f'{i+1}/80 {f.result()}',flush=True)
 records=[json.loads(p.read_text()) for p in args.output.glob('*.json')]
 report=summarize_grid(records);report['identity']=identity
 (LAB/'_paper_l088_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report['selected'])
if __name__=='__main__':main()

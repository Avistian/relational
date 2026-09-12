"""Fresh L065 author experiment; historical _verify_l065.py is preserved."""
import argparse,concurrent.futures,json
from pathlib import Path
import torch
from relkit import query_embeddings_l065_v2 as core
ROOT=Path(__file__).resolve().parent

def run_one(dataset,seeds):
 torch.set_num_threads(1);model,_=core.load_pretrained(core.ensure_checkpoint(ROOT))
 return core.run_experiment(ROOT,model,dict(datasets=[dataset],seeds=seeds))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=core.PRESETS65,default='closer');p.add_argument('--output',type=Path,default=ROOT/'_verify_l065_v2_results.json');a=p.parse_args();config=core.PRESETS65[a.preset]
 if a.output.exists():p.error('Refusing to overwrite existing evidence; supply a fresh --output path')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  results=list(pool.map(run_one,config['datasets'],[config['seeds']]*len(config['datasets'])))
 result=results[0];assert all(r['kernel_identity']['sha256']==result['kernel_identity']['sha256'] for r in results)
 result['records']=[record for r in results for record in r['records']];result['config']=config
 with a.output.open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')

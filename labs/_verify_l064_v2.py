"""Fresh bounded L064 full-checkpoint inference; never reads historical scores."""
import argparse,hashlib,importlib.metadata,json,platform
from pathlib import Path
import torch
from relkit import tabpfn_l064_v2 as core
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=list(core.PRESETS),default='closer');p.add_argument('--output',type=Path,default=ROOT/'_verify_l064_v2_results.json');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Fresh output required; no resume')
 torch.set_num_threads(1);model,checkpoint_config=core.load_pretrained(core.ensure_checkpoint(ROOT));result=core.run_experiment(ROOT,model,core.PRESETS[a.preset]);result['checkpoint_config']=checkpoint_config
 result['source_sha256']=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest();result['versions']={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn','pandas']};result['hardware']=platform.machine()+' CPU; one torch thread'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')

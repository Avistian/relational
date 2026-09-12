"""Fresh original-checkpoint context-size experiment; no overwrite or resume."""
import argparse,json
from pathlib import Path
import torch
from relkit import tabicl_l066_v2 as core
ROOT=Path(__file__).resolve().parent

def run(preset='closer',output=None):
 output=Path(output) if output else ROOT/'_verify_l066_v2_results.json'
 if output.exists():raise FileExistsError('Choose a fresh output path; no overwrite/resume')
 if preset=='paper':raise ValueError('Original synthetic pretraining and TALENT benchmark are not implemented by this numeric inference operator')
 torch.set_num_threads(1);model=core.load_pretrained(core.ensure_checkpoint(ROOT));result=core.run_experiment(ROOT,model,core.PRESETS66[preset])
 result['source_sha256']=__import__('hashlib').sha256(Path(core.__file__).read_bytes()).hexdigest()
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return output
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',default='closer',choices=['smoke','lab','closer','paper']);p.add_argument('--output');a=p.parse_args();print(run(a.preset,a.output))

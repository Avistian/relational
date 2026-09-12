"""Fresh full pretrained LoCalPFN adaptation. Refuse overwrite; historical evidence untouched."""
import argparse,json,hashlib
from pathlib import Path
import torch
from relkit import localpfn_l067_v2 as core
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','lab','closer','paper'],default='closer');p.add_argument('--output',required=True);a=p.parse_args()
 output=Path(a.output)
 if output.exists():raise FileExistsError(output)
 if a.preset=='paper':raise ValueError('The 95-dataset TabZilla 10-fold full-budget benchmark is not implemented by this numeric subset operator')
 torch.set_num_threads(1);model,_=core.load_pretrained(core.ensure_checkpoint(ROOT));result=core.run_experiment(ROOT,model,core.PRESETS67[a.preset])
 result['operator_reference_sha256']=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest();output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(output)

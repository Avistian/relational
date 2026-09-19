"""Run the complete fixed-split Cora experiment; no test-driven tuning."""
import argparse,json,hashlib
from pathlib import Path
from relkit.message_passing import run_cora
p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--output',default='_paper_l078_results.json');a=p.parse_args()
if a.seeds<1:p.error('seeds must be positive')
root=Path(__file__).resolve().parent
r=run_cora(a.seeds,root);r['implementation_sha256']=hashlib.sha256((root/'relkit/message_passing.py').read_bytes()).hexdigest();r['source_manifest_sha256']=hashlib.sha256((root/'_sources_l078.json').read_bytes()).hexdigest()
out=Path(a.output);out=out if out.is_absolute() else root/out;out.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='runs'})

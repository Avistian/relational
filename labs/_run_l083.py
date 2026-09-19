"""Full PPI supervised mean experiment, all three published learning rates; no test selection."""
import argparse,hashlib,json
from pathlib import Path
from relkit.graphsage_l083 import *
LAB=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);p.add_argument('--preset',choices=['paper','smoke'],default='paper');p.add_argument('--output');a=p.parse_args()
    torch.set_num_threads(1)
    manifest=json.loads((LAB/'_sources_l083.json').read_text());data=load_ppi(LAB/'data/l083',manifest)
    config=copy.deepcopy(PAPER_CONFIG)
    if a.preset=='smoke':config.update(epochs=1,branch_dim=16,learning_rates=[.01])
    out=Path(a.output) if a.output else LAB/f'_{a.preset}_l083_results.json'
    result=experiment(data,list(range(123,123+a.seeds)),config,out)
    result.update(implementation_sha256=hashlib.sha256((LAB/'relkit/graphsage_l083.py').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((LAB/'_sources_l083.json').read_bytes()).hexdigest(),original_tensorflow='NOT_RUN')
    out.write_text(json.dumps(result,indent=2)+'\n');print(result['mean'],result['sample_sd'])

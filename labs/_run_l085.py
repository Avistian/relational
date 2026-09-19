"""Fresh, deterministic L085 runs; no cached scores or resume behavior."""
import argparse,json,hashlib,platform
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from oversmoothing_l085 import karate_experiment,train_depth
from gcn_l082 import load_cora

def run(lane,seeds,output):
    torch.set_num_threads(1);root=Path(__file__).resolve().parent
    manifest=json.loads((root/'_sources_l085.json').read_text())
    for record in manifest['files']:
        p=root/'sources/l085'/record['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Source identity mismatch: '+str(p))
    if hashlib.sha256((root/manifest['graph']['path']).read_bytes()).hexdigest()!=manifest['graph']['sha256']:
        raise ValueError('Karate graph identity mismatch')
    if lane=='karate':
        graph=json.loads((root/'sources/l085/karate.json').read_text());result=karate_experiment(graph,seeds)
    else:
        data=load_cora(root);runs=[]
        for depth in [1,2,4,8,16]:
            for seed in range(seeds):
                row=train_depth(data,depth,seed);runs.append(row)
                print(depth,seed,row['test_accuracy'],row['epochs'],flush=True)
        result={'experiment':'Cora fixed-split depth diagnostic; extension, not Li et al. paper figure or classification tables','runs':runs,'status':'LOCAL_EXTENSION','depths':[1,2,4,8,16]}
    result['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}
    result['implementation_sha256']=hashlib.sha256((root/'relkit/oversmoothing_l085.py').read_bytes()).hexdigest()
    result['source_manifest_sha256']=hashlib.sha256((root/'_sources_l085.json').read_bytes()).hexdigest()
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print('Wrote',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=['karate','cora'],default='karate');p.add_argument('--seeds',type=int,default=100);p.add_argument('--output',required=True);a=p.parse_args()
    if a.seeds<1:p.error('seeds must be positive')
    run(a.lane,a.seeds,a.output)

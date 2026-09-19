"""Run full released-data baselines and separately labeled GCN teaching experiments."""
import argparse,hashlib,json,sys,importlib.metadata as md
from pathlib import Path
import numpy as np
from relkit.link_l087 import run_paper,load_graph,train_link
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['paper','teaching','all'],default='all');p.add_argument('--output',type=Path,default=ROOT/'results/l087');args=p.parse_args()
    manifest=json.loads((ROOT/'_sources_l087.json').read_text());args.output.mkdir(parents=True,exist_ok=True)
    environment={'python':sys.version,'packages':{n:md.version(n) for n in ['numpy','scipy','torch','scikit-learn']},'implementation_sha256':hashlib.sha256((ROOT/'relkit/link_l087.py').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((ROOT/'_sources_l087.json').read_bytes()).hexdigest()}
    if args.track in ['paper','all']:
        r=run_paper(manifest,ROOT/'sources/l087',args.output/'paper');r['environment']=environment
        (ROOT/'_paper_l087_results.json').write_text(json.dumps(r,indent=2)+'\n')
    if args.track in ['teaching','all']:
        e,n=load_graph('USAir',manifest,ROOT/'sources/l087');runs=[]
        for seed in [87,88,89]:
            r,a=train_link(e,n,seed);fp=args.output/f'teaching-{seed}.npz';np.savez_compressed(fp,**a);r['artifact_sha256']=hashlib.sha256(fp.read_bytes()).hexdigest();runs.append(r);print(seed,r['test_auc'],r['ranking']['mrr'],flush=True)
        (ROOT/'_teaching_l087_results.json').write_text(json.dumps({'runs':runs,'environment':environment},indent=2)+'\n')

"""Named published experiment lanes, separate from the ACM course comparison."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
a=argparse.ArgumentParser();a.add_argument('--target',choices=['aifb','hgt-cs'],required=True);a.add_argument('--device',default='cpu');a.add_argument('--with-rgcn',action='store_true',help='Also run the course R-GCN comparator; not the original paper baseline');args=a.parse_args()
if args.target=='aifb':
 from rgcn_l091 import run_aifb
 r=run_aifb(P/'data/l091',json.loads((P/'_sources_l091.json').read_text()),seeds=range(10),bases=0)
 r['scope']='Fresh L099 replay of complete AIFB release-protocol port; not the ACM comparison'
 r['implementation_sha256']=hashlib.sha256((P/'relkit/rgcn_l091.py').read_bytes()).hexdigest()
 (P/'_paper_l099_aifb_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r['mean'],r['sample_sd'])
else:
 m=json.loads((P/'_sources_l093.json').read_text())
 subprocess.run([sys.executable,str(P/'_run_l093.py'),'--preset','paper','--data',str(P/'data/l093/graph_CS.pk'),'--sha256',m['cs_data']['sha256'],'--seeds','0','1','2','3','4','--arms',*(['hgt','rgcn'] if args.with_rgcn else ['hgt']),'--device',args.device,'--output',str(P/'_paper_l099_hgt_cs_results.json')],check=True)

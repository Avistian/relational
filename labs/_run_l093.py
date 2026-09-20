"""OAG runner; paper/release require CS bytes, teaching/smoke use NN separately."""
import argparse,json,sys,platform
from pathlib import Path
import numpy as np,torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hgt_l093 import load_oag,protocol_config,train_oag,file_sha256
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--preset',choices=['smoke','teaching','paper','release'],default='smoke');parser.add_argument('--data',type=Path);parser.add_argument('--sha256');parser.add_argument('--seeds',type=int,nargs='+');parser.add_argument('--arms',nargs='+',choices=['hgt','hgt_no_rte','rgcn'],default=['hgt']);parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,default=P/'_smoke_l093_results.json');a=parser.parse_args()
m=json.loads((P/'_sources_l093.json').read_text());large=a.preset in ('paper','release')
if large and (not a.data or not a.sha256):parser.error('Named target requires --data /path/graph_CS.pk --sha256 VERIFIED_CS_HASH. NN cannot substitute for CS.')
path=a.data or P/'data/l093/graph_NN.pk';digest=a.sha256 or m['nn_data']['sha256']
expected=m['cs_data']['sha256'] if large else m['nn_data']['sha256']
if digest!=expected:parser.error('Dataset hash does not match the named preset dataset; do not relabel another graph')
if large and path.name!='graph_CS.pk':parser.error('Paper/release named target expects graph_CS.pk; smaller graphs belong to teaching/smoke')
seeds=a.seeds or ([0,1,2,3,4] if large else ([0,1,2] if a.preset=='teaching' else [0]))
config=protocol_config(a.preset);config['device']=a.device;torch.set_num_threads(2)
result={'status':'RUNNING','preset':a.preset,'dataset':'CS' if large else 'NN','data_sha256':digest,'source_revision':m['source_revision'],'full_paper_parity':'NOT_ESTABLISHED','historical_parity':'INCOMPARABLE','scope':'Named target reconstruction' if large else 'Teaching only; not a paper-table result','environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':a.device},'runs':[],'planned_runs':len(seeds)*len(a.arms)}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
graph=load_oag(path,digest)
result['implementation_sha256']=file_sha256(P/'relkit/hgt_l093.py')
for seed in seeds:
 for arm in a.arms:
  row,_=train_oag(graph,seed,config,arm,progress=True);result['runs'].append(row);a.output.write_text(json.dumps(result,indent=2)+'\n')
result['summary']={arm:{metric:{'mean':float(np.mean(v:=[r['test_'+metric] for r in result['runs'] if r['arm']==arm])),'sample_sd':float(np.std(v,ddof=1)) if len(v)>1 else None} for metric in ('ndcg','mrr')} for arm in a.arms}
result['status']='COMPLETE';result['implementation_sha256']=file_sha256(P/'relkit/hgt_l093.py');a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['summary'])

"""Unmodified publication-day RelBench replay. Preflight does not train.

Run inside the isolated environment documented in l076-reproduction.md.
Each run records exact source, configuration, packages, stdout and predictions.
"""
import argparse,ast,hashlib,importlib.metadata,json,platform,runpy,subprocess,sys
from pathlib import Path
PIN='5894184f3d1b2432feb9a208a8aaf18b106fbdf4'
LAB=Path(__file__).resolve().parent

def preflight():
 versions={};problems=[]
 for name,expected in [('torch','2.3.0'),('pytorch-frame','0.2.3'),('torch-geometric','2.5.3'),('pyg-lib','0.4.0'),('sentence-transformers','3.0.1')]:
  try:versions[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:versions[name]='MISSING'
  if versions[name].split('+')[0]!=expected:problems.append(f'{name}: need {expected}, found {versions[name]}')
 try:
  import pyg_lib
  import torch_geometric.typing as typing
  if not typing.WITH_EDGE_TIME_NEIGHBOR_SAMPLE:problems.append('Temporal edge sampling support absent')
 except ImportError:problems.append('Cannot import pyg_lib temporal sampling backend')
 return {'status':'BLOCKED' if problems else 'PASS','machine':platform.machine(),'python':sys.version,'versions':versions,'problems':problems,'benchmark':'NOT_RUN'}

def prepare(folder):
 if not (folder/'.git').exists():subprocess.run(['git','clone','https://github.com/snap-stanford/relbench.git',str(folder)],check=True)
 actual=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
 if actual!=PIN:
  if subprocess.check_output(['git','-C',str(folder),'status','--porcelain'],text=True).strip():raise RuntimeError('Refuse to overwrite modified checkout')
  subprocess.run(['git','-C',str(folder),'checkout','--detach',PIN],check=True)
 manifest=json.loads((LAB/'_sources_l076.json').read_text())
 for path,item in manifest['files'].items():
  relative=path.split('sources/l076-relbench/',1)[1]
  assert hashlib.sha256((folder/relative).read_bytes()).hexdigest()==item['sha256'],relative
 return folder

def command(folder,seed,cache):
 return [sys.executable,str(folder/'examples/gnn_node.py'),'--dataset','rel-f1','--task','driver-dnf',
 '--lr','0.005','--epochs','10','--batch_size','512','--channels','128','--aggr','sum',
 '--num_layers','2','--num_neighbors','128','--temporal_strategy','uniform',
 '--max_steps_per_epoch','2000','--num_workers','0','--seed',str(seed),'--cache_dir',str(cache)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--run',action='store_true')
 p.add_argument('--checkout',type=Path,default=Path('/tmp/l076-relbench-replay'));p.add_argument('--output',type=Path,default=Path('l076-replay'))
 p.add_argument('--seeds',type=int,nargs='+',default=list(range(5)));args=p.parse_args()
 report=preflight();print(json.dumps(report,indent=2))
 args.output.mkdir(parents=True,exist_ok=True);(args.output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
 if not args.run:return
 if report['status']!='PASS':raise SystemExit('Historical environment preflight failed; no training started')
 folder=prepare(args.checkout.resolve());out=args.output.resolve()
 # Use the pinned source package, not an installed current relbench.
 import os
 env=dict(os.environ,PYTHONPATH=str(folder),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 results=[]
 for seed in args.seeds:
  cmd=command(folder,seed,out/'cache');log=out/f'seed-{seed}.log'
  # runpy executes the exact released trainer, then archives its final prediction arrays.
  wrapper='''import runpy,sys,json,numpy as np
from pathlib import Path
script,output,*args=sys.argv[1:]
sys.path.insert(0,str(Path(script).parent));sys.argv=[script]+args
state=runpy.run_path(script,run_name="__main__")
np.savez(output,val_pred=state["val_pred"],test_pred=state["test_pred"])
'''
  invocation=[sys.executable,'-c',wrapper,cmd[1],str(out/f'seed-{seed}-predictions.npz')]+cmd[2:]
  with log.open('w') as f:subprocess.run(invocation,cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  metrics={}
  for line in log.read_text().splitlines():
   for prefix,key in [('Best Val metrics: ','val'),('Best test metrics: ','test')]:
    if line.startswith(prefix):metrics[key]=ast.literal_eval(line[len(prefix):])
  if set(metrics)!={'val','test'}:raise RuntimeError('Missing final metrics in '+str(log))
  results.append({'seed':seed,'command':cmd,'metrics':metrics})
  (out/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
 import numpy as np
 values=[r['metrics']['test']['roc_auc'] for r in results]
 data_files=[]
 for base in [out/'cache',Path.home()/'.cache/relbench']:
  if base.exists():
   for f in sorted(base.rglob('*')):
    if f.is_file():data_files.append({'path':str(f),'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()})
 summary={'status':'EXECUTED_RELEASE_REPLAY','commit':PIN,'seeds':args.seeds,'test_auroc_mean':float(np.mean(values)),
 'test_auroc_sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'paper_target_auroc':.7262,
 'comparison':'INCOMPARABLE_PENDING_PROTOCOL_AUDIT','environment':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
 'data_files':data_files,'deviations':['Chosen seeds0-4; original seed IDs unidentified','Reconstructed environment, not author lock','Released fanouts128,64 retained','External GloVe checkpoint is upstream-selected; cache/hash must be audited']}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

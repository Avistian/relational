"""Bounded OGB reproduction: pilot then ten complete 500-epoch fits."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l112-ogb-gcn')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install_from_requirements(ROOT/'labs/requirements-l112-runtime.txt')
 .add_local_file(ROOT/'labs/relkit/ogb_l112.py','/work/relkit/ogb_l112.py')
 .add_local_file(ROOT/'labs/_source_check_l112.py','/work/_source_check_l112.py')
 .add_local_dir(ROOT/'labs/sources/l112','/work/sources/l112')
 .add_local_file(ROOT/'labs/data/l112/arxiv.zip','/data/arxiv.zip'))
volume=modal.Volume.from_name('l112-ogb-evidence',create_if_missing=True)
RATE=.00020796 # T4 + 2 CPU cores + 8 GiB, verified 2026-09-26

def run(seed,epochs,preset):
 import sys,json,time,torch,hashlib,platform
 sys.path.insert(0,'/work')
 from relkit.ogb_l112 import load_arxiv,normalized_adjacency,train_run
 from _source_check_l112 import check
 torch.set_num_threads(1);start=time.perf_counter()
 digest=hashlib.sha256(Path('/work/relkit/ogb_l112.py').read_bytes()).hexdigest()
 root=Path('/evidence')/digest/preset/f'seed-{seed}';root.mkdir(parents=True,exist_ok=True)
 try:
  if preset=='pilot':(root/'source_parity.json').write_text(json.dumps(check('cuda'),indent=2))
  x,edge,y,split,audit=load_arxiv('/data');adj=normalized_adjacency(edge,len(y))
  identity={'source_sha256':digest,'data':audit,'torch':torch.__version__,'python':sys.version,'device':torch.cuda.get_device_name(),'platform':platform.platform(),'seed':seed,'epochs':epochs,'preset':preset,'normalized_nonzero':adj.values().numel()}
  (root/'identity.json').write_text(json.dumps(identity,indent=2))
  result=train_run(x,adj,y,split,seed,epochs,root,device='cuda')
  elapsed=time.perf_counter()-start
  done={'status':'COMPLETE','seconds':elapsed,'resource_usd':elapsed*RATE,'training_seconds':result['seconds'],'path':str(root),'scores':result['scores']}
  (root/'completed.json').write_text(json.dumps(done,indent=2));return done
 finally:volume.commit()
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=600,retries=0,volumes={'/evidence':volume})
def pilot():return run(100,10,'pilot')
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=600,retries=0,volumes={'/evidence':volume})
def full(seed):return run(seed,500,'paper')
@app.local_entrypoint()
def main(mode:str='pilot'):
 if mode=='pilot':print(pilot.remote())
 elif mode=='paper':
  import json,hashlib
  budget=json.loads((ROOT/'labs/_budget_l112.json').read_text())
  assert budget['pilot_approved_for_full'] and budget['source_sha256']==hashlib.sha256((ROOT/'labs/relkit/ogb_l112.py').read_bytes()).hexdigest()
  for result in full.map(range(10)):print(result)
 else:raise ValueError(mode)

"""USD10 aggregate budget; no automatic retries. Prepare, pilot, then gated complete fits."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l113-ogb-scaling')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install_from_requirements(ROOT/'labs/requirements-l113-runtime.txt')
 .pip_install('pyg-lib==0.4.0','torch-scatter==2.1.2','torch-sparse==0.6.18',find_links='https://data.pyg.org/whl/torch-2.8.0+cu128.html',extra_options='--only-binary=:all:')
 .add_local_file(ROOT/'labs/relkit/scaling_l113.py','/work/relkit/scaling_l113.py')
 .add_local_file(ROOT/'labs/_prepare_l113.py','/work/_prepare_l113.py')
 .add_local_file(ROOT/'labs/_audit_l113.py','/work/_audit_l113.py')
 .add_local_file(ROOT/'labs/_replay_l113.py','/work/_replay_l113.py')
 .add_local_file(ROOT/'labs/_check_l113.py','/work/_check_l113.py')
 .add_local_file(ROOT/'labs/_source_check_l113.py','/work/_source_check_l113.py')
 .add_local_dir(ROOT/'labs/sources/l113','/work/sources/l113'))
volume=modal.Volume.from_name('l113-products',create_if_missing=True)
RATE=.00028744 # T4 + 4 physical CPU + 32 GiB at modal.com/pricing 2026-09-26
@app.function(image=image,cpu=4,memory=65536,timeout=1800,retries=0,volumes={'/persist':volume})
def prepare():
 import sys,torch
 sys.path.insert(0,'/work');torch.set_num_threads(4)
 from _prepare_l113 import prepare
 try:return prepare('/persist/data')
 finally:volume.commit()

@app.function(image=image,cpu=4,memory=65536,timeout=600,retries=0,volumes={'/persist':volume})
def audit():
 import sys,torch
 sys.path.insert(0,'/work');torch.set_num_threads(4)
 from _audit_l113 import audit
 try:return audit('/persist/data')
 finally:volume.commit()

def run(seed,epochs,preset,max_seconds):
 import sys,json,time,torch,hashlib,importlib.metadata
 sys.path.insert(0,'/work');torch.set_num_threads(4);start=time.perf_counter()
 from relkit.scaling_l113 import train_run
 from _check_l113 import check_model
 from relkit import scaling_l113 as m
 check_model(m)
 from _source_check_l113 import check
 parity=check('cuda')
 digest=hashlib.sha256(Path('/work/relkit/scaling_l113.py').read_bytes()).hexdigest()
 root=Path('/persist')/digest/preset/f'seed-{seed}'
 blob=torch.load('/persist/data/data.pt',weights_only=False)
 clusters=torch.load('/persist/data/clusters.pt',weights_only=False)
 adj=torch.load('/persist/data/adj.pt',weights_only=False)
 try:
  result=train_run(blob['data'],clusters,adj,blob['split'],root,seed,epochs,'cuda',max_seconds,workers=0)
  done={'status':result['status'],'source_sha256':digest,'seconds':time.perf_counter()-start,'resource_usd':(time.perf_counter()-start)*RATE,
        'source_parity':parity,'data':json.loads(Path('/persist/data/prepared.json').read_text()),'versions':{k:importlib.metadata.version(k) for k in ['torch','torch-geometric','pyg-lib','torch-sparse','ogb']},'gpu':torch.cuda.get_device_name(),'path':str(root)}
  (root/'identity.json').write_text(json.dumps(done,indent=2));return done
 finally:volume.commit()
@app.function(image=image,gpu='T4',cpu=4,memory=32768,timeout=900,retries=0,volumes={'/persist':volume})
def pilot():return run(100,2,'pilot',720)
@app.function(image=image,gpu='T4',cpu=4,memory=32768,timeout=2500,retries=0,volumes={'/persist':volume})
def full(seed):return run(seed,50,'paper',2350)
@app.function(image=image,gpu='T4',cpu=4,memory=32768,timeout=900,retries=0,volumes={'/persist':volume})
def replay(preset: str="paper"):
 import sys,torch,hashlib
 sys.path.insert(0,'/work');torch.set_num_threads(4)
 from _replay_l113 import replay
 digest=hashlib.sha256(Path('/work/relkit/scaling_l113.py').read_bytes()).hexdigest()
 try:return replay('/persist',digest,[100] if preset=='pilot' else range(10),preset)
 finally:volume.commit()
@app.local_entrypoint()
def main(mode:str='prepare',seed:int=0):
 if mode=='prepare':print(prepare.remote())
 elif mode=='pilot':print(pilot.remote())
 elif mode=='audit':print(audit.remote())
 elif mode=='replay':print(replay.remote())
 elif mode=='replay_pilot':print(replay.remote('pilot'))
 elif mode in ('paper','paper_all'):
  import json,hashlib
  b=json.loads((ROOT/'labs/_budget_l113.json').read_text())
  assert b['pilot_approved_for_full'] and seed in b['authorized_seeds']
  assert b['source_sha256']==hashlib.sha256((ROOT/'labs/relkit/scaling_l113.py').read_bytes()).hexdigest()
  if mode=='paper_all':
   assert b['authorized_seeds']==list(range(10))
   for result in full.map(range(10)):print(result)
  else:print(full.remote(seed))
 else:raise ValueError(mode)

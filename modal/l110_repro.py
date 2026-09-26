"""Bounded fresh paired training; at most 600 + 10*3500 aggregate GPU seconds."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l110-temporal-checkpoint')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.3.2','scikit-learn==1.7.1')
 .add_local_file(ROOT/'labs/relkit/checkpoint_l110.py','/work/relkit/checkpoint_l110.py')
 .add_local_file(ROOT/'labs/relkit/tgn_l102.py','/work/relkit/tgn_l102.py')
 .add_local_file(ROOT/'labs/_source_check_l110.py','/work/_source_check_l110.py')
 .add_local_file(ROOT/'labs/_replay_source_l110.py','/work/_replay_source_l110.py')
 .add_local_dir(ROOT/'labs/sources/l102','/work/sources/l102'))
volume=modal.Volume.from_name('l110-checkpoint-evidence',create_if_missing=True)
def run(seed,epochs,preset):
 import sys,json,time,hashlib,subprocess,torch,numpy,platform
 sys.path.insert(0,'/work')
 from relkit.checkpoint_l110 import load_wikipedia,run_training
 torch.set_num_threads(1);start=time.perf_counter()
 source=hashlib.sha256(Path('/work/relkit/checkpoint_l110.py').read_bytes()).hexdigest()
 root=Path('/evidence')/source/preset/f'seed-{seed}';root.mkdir(parents=True,exist_ok=True)
 nodes,edges,data,audit=load_wikipedia(root/'data')
 identity={'source_sha256':source,'audit':audit,'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'device':torch.cuda.get_device_name(),'platform':platform.platform(),'seed':seed,'epochs':epochs,'preset':preset}
 (root/'identity.json').write_text(json.dumps(identity,indent=2))
 try:
  if preset=='pilot':
   subprocess.run([sys.executable,'/work/_source_check_l110.py','cuda'],check=True)
   (root/'source_parity.json').write_bytes(Path('/work/_source_check_l110_results.json').read_bytes())
  for arm in ['release','clean']:
   run_training(nodes,edges,data,seed=seed,epochs=epochs,output=root/arm,device='cuda',arm=arm)
   volume.commit()
  if preset=='paper' and seed==0:
   from _replay_source_l110 import replay
   replay(root/'release',root/'data','cuda',seed)
  elapsed=time.perf_counter()-start
  result={'status':'COMPLETE','seconds':elapsed,'resource_cost_usd':elapsed*.00020796,'path':str(root)}
  (root/'completed.json').write_text(json.dumps(result,indent=2));return result
 finally:volume.commit()
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=600,retries=0,volumes={'/evidence':volume})
def pilot():return run(0,1,'pilot')
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=3500,retries=0,volumes={'/evidence':volume})
def paired(seed):return run(seed,50,'paper')
@app.local_entrypoint()
def main(mode:str='pilot'):
 if mode=='pilot':print(pilot.remote())
 elif mode=='paper':
  for result in paired.map(range(10)):print(result)
 else:raise ValueError(mode)

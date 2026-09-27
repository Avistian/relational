"""USD10 aggregate, no automatic retries; preprocessing then measured pilot gate."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l132-identity-aware')
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt')
 .pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html')
 .add_local_file(ROOT/'labs/_run_l132.py','/work/_run_l132.py')
 .add_local_file(ROOT/'labs/relkit/identity_l132.py','/work/relkit/identity_l132.py')
 .add_local_dir(ROOT/'labs/sources/l132','/work/sources/l132')
 .add_local_file(ROOT/'labs/sources/l117/text_model.json','/work/sources/l117/text_model.json'))
volume=modal.Volume.from_name('l132-identity-evidence',create_if_missing=True)
RATE=.00026124
@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=3600,retries=0,volumes={'/evidence':volume})
def prepare_worker():
 import sys,time,json,traceback,os
 os.environ.update(XDG_CACHE_HOME="/evidence/cache",HF_HOME="/evidence/hf")
 sys.path.insert(0,'/work');from _run_l132 import prepare
 start=time.perf_counter()
 try:return prepare('/evidence/prepared')
 except Exception:
  Path('/evidence/preparation-error.txt').write_text(traceback.format_exc());raise
 finally:
  Path('/evidence/preparation-cost.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'resource_usd':(time.perf_counter()-start)*RATE}));volume.commit()
@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=1800,retries=0,volumes={'/evidence':volume})
def worker(variant,seed,epochs,mode):
 import sys,time,json,traceback,uuid,os
 os.environ.update(XDG_CACHE_HOME="/evidence/cache",HF_HOME="/evidence/hf")
 sys.path.insert(0,'/work');from _run_l132 import run
 output=Path('/evidence')/mode/f'{variant}-{seed}';output.mkdir(parents=True,exist_ok=True)
 assert not (output/'completed.json').exists()
 start=time.perf_counter();status='FAILED'
 try:
  result=run(variant,seed,epochs,'/evidence/prepared',output);status='COMPLETE';return result
 except Exception:
  (output/'error.txt').write_text(traceback.format_exc());raise
 finally:
  seconds=time.perf_counter()-start
  (output/'completed.json').write_text(json.dumps(dict(status=status,seconds=seconds,resource_usd=seconds*RATE,run_uuid=str(uuid.uuid4()),variant=variant,seed=seed,epochs=epochs),indent=2));volume.commit()
@app.local_entrypoint()
def main(mode:str='prepare'):
 import json,fcntl,datetime,hashlib
 path=ROOT/'labs/_budget_l132.json'
 with path.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  assert mode in ['prepare','pilot','paper']
  assert not any(x['mode']==mode for x in b['reservations'])
  for rel,digest in b['source_hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
  seconds=3600 if mode=='prepare' else (3600 if mode=='pilot' else 18000)
  assert sum(x['reserved_seconds'] for x in b['reservations'])+seconds<=28800
  if mode=='paper':assert b['pilot_approved_for_full']
  b['reservations'].append(dict(mode=mode,reserved_seconds=seconds,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 if mode=='prepare':print(prepare_worker.remote())
 else:
  calls=[(v,s,1 if mode=='pilot' else 20,mode) for v in ['sage','idgnn'] for s in ([100] if mode=='pilot' else range(5))]
  for r in worker.starmap(calls,return_exceptions=True):print(r)

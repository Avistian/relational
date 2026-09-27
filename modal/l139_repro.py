"""Shared USD10 guard: every attempt reserves its worst-case worker cost."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l139-trial-reproduction')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
image=image.add_local_dir(ROOT/'labs/sources/l139','/work/sources/l139').add_local_file(ROOT/'labs/_prepare_l139.py','/work/_prepare_l139.py').add_local_file(ROOT/'labs/_full_l139.py','/work/_full_l139.py').add_local_file(ROOT/'labs/relkit/trial_l139.py','/work/relkit/trial_l139.py').add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py')
volume=modal.Volume.from_name('l139-trial-evidence',create_if_missing=True);prior=modal.Volume.from_name('l132-identity-evidence')
RATE=.000164+2*.0000131+64*.00000222
@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=1800,retries=0,volumes={'/evidence':volume,'/prior':prior})
def worker(mode,seed=0):
 import sys,time,json,traceback,os
 sys.path.insert(0,'/work');os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 root=Path('/evidence');start=time.perf_counter();name=mode if mode.startswith('prepare') else f'{mode}/seed-{seed}';cost=root/(name.replace('/','-')+'-cost.json')
 # Refuse infrastructure retries too: the attempt marker is committed first.
 marker=root/(name.replace('/','-')+'-started.json')
 assert not marker.exists(),'Attempt already started; no automatic redispatch'
 marker.write_text(json.dumps({'started':time.time()}));volume.commit()
 try:
  if mode.startswith('prepare'):
   from _prepare_l139 import prepare
   return prepare(root,Path('/prior'))
  from _full_l139 import full_run
  return full_run(root/name,seed=seed,epochs=1 if mode.startswith('pilot') else 20,prepared_root=root)
 except Exception:
  (root/(name.replace('/','-')+'-error.txt')).write_text(traceback.format_exc());raise
 finally:
  cost.write_text(json.dumps(dict(seconds=time.perf_counter()-start,resource_usd=(time.perf_counter()-start)*RATE)));volume.commit()
@app.local_entrypoint()
def main(mode:str='prepare'):
 import json,hashlib,fcntl,datetime
 assert mode in ['prepare','prepare2','pilot','pilot2','full']
 seeds=list(range(5)) if mode=='full' else [99 if mode.startswith('pilot') else 0]
 if mode=='full':
  pilot=json.loads((ROOT/'labs/evidence/l139/pilot2/seed-99/result.json').read_text())
  projected=pilot['seconds']+19*pilot['history'][0]['seconds']
  assert 1.3*projected+120<1800,('Full fit exceeds reservation',projected)
 budget=ROOT/'labs/_budget_l139.json'
 with budget.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);assert not any(r['phase']==mode for r in b['reservations'])
  upper=len(seeds)*1800*RATE
  assert sum(r['upper_usd'] for r in b['reservations'])+upper+b['overhead_reserve_usd']<=b['budget_usd']
  b['reservations'].append(dict(phase=mode,seeds=seeds,timeout_per_worker=1800,upper_usd=upper,rate=RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'labs/_prepare_l139.py',ROOT/'labs/_full_l139.py',ROOT/'labs/relkit/rdl_l117.py',ROOT/'modal/l139_repro.py']}))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 # Sequential fits bound memory pressure and ease complete artifact collection.
 for seed in seeds:print(worker.remote(mode,seed))

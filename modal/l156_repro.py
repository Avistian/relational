"""Approved fresh F1 experiment: immutable source-pinned aggregate reservations."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l156-regression-portfolio');RATE=.00022572
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
FILES=['_run_l156.py','relkit/temporal_audit_l156.py','_run_l152.py','_run_l117.py','relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/regression_l152.py']
for p in FILES:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
image=image.add_local_dir(ROOT/'labs/sources/l117','/work/sources/l117')
volume=modal.Volume.from_name('l156-regression-evidence',create_if_missing=True)

def reserve(phase,workers=1,seconds=1800):
 import json,fcntl,hashlib,datetime
 with (ROOT/'labs/_budget_l156.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  assert not any(r['phase']==phase for r in b['reservations']),'Attempt already reserved'
  for p,h in b['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
  assert workers+sum(r['workers'] for r in b['reservations'])<=12,'Twelve-slot cap'
  upper=workers*seconds*RATE
  assert upper+sum(r['upper_usd'] for r in b['reservations'])+b['overhead_reserve_usd']<=b['budget_usd'],'USD10 cap'
  b['reservations'].append(dict(phase=phase,workers=workers,seconds=seconds,upper_usd=upper,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0,volumes={'/evidence':volume})
def worker(seed,lane='released'):
 import os,sys,time,json,hashlib,uuid,traceback
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path.insert(0,'/work')
 from _run_l156 import run
 root=Path('/evidence')/('paper' if lane=='released' else lane)/f'seed-{seed}';root.mkdir(parents=True,exist_ok=False)
 (root/'started.json').write_text(json.dumps(dict(uuid=str(uuid.uuid4()),seed=seed)));volume.commit();start=time.perf_counter()
 try:
  r=run(seed,10,root,lane);done=dict(status='COMPLETE',seed=seed,epochs=10,scores=r['scores'],seconds=time.perf_counter()-start,lesson=156,lane=lane,source_hashes={p:hashlib.sha256((Path('/work')/p).read_bytes()).hexdigest() for p in FILES})
  (root/'completed.json').write_text(json.dumps(done,indent=2));return done
 except Exception:
  (root/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  elapsed=time.perf_counter()-start;(root/'cost.json').write_text(json.dumps(dict(seconds=elapsed,worker_body_usd=elapsed*RATE)));volume.commit()

@app.local_entrypoint()
def main(phase:str='pilot'):
 import json
 assert json.loads((ROOT/'labs/evidence/l156/preflight.json').read_text())['status']=='PASS'
 if phase=='pilot':reserve(phase);print(worker.remote(0))
 elif phase=='remaining':
  pilot=json.loads((ROOT/'labs/evidence/l156/paper/seed-0/completed.json').read_text());assert pilot['status']=='COMPLETE' and pilot['seconds']*1.25+120<1800
  reserve(phase,4)
  for r in worker.map(range(1,5)):print(r)
 elif phase=='correction':
  assert json.loads((ROOT/'labs/evidence/l156/correction-protocol.json').read_text())['frozen_before_corrected_scores']
  reserve(phase,5)
  for r in worker.starmap([(i,'fit_horizon') for i in range(5)]):print(r)
 else:raise ValueError(phase)

"""USD10 aggregate, immutable attempts, no automatic retries."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l141-relgnn')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
for p in ['_full_l141.py','_parity_l141.py','_check_l141.py','relkit/relgnn_l141.py']:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
image=image.add_local_dir(ROOT/'labs/sources/l141','/work/sources/l141')
volume=modal.Volume.from_name('l141-relgnn-evidence',create_if_missing=True)
RATE=.00022572

def reserve(phase,workers=1,seconds=1800):
 import json,fcntl,hashlib,datetime
 with (ROOT/'labs/_budget_l141.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  assert not any(x['phase']==phase for x in b['reservations']),'Duplicate reservation'
  upper=workers*seconds*RATE
  assert sum(x['upper_usd'] for x in b['reservations'])+upper+b['overhead_reserve_usd']<=10,'Aggregate cap'
  b['reservations'].append(dict(phase=phase,workers=workers,timeout=seconds,upper_usd=upper,rate=RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={p:hashlib.sha256((ROOT/'labs'/p).read_bytes()).hexdigest() for p in ['_full_l141.py','relkit/relgnn_l141.py']}))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0,volumes={'/evidence':volume})
def worker(phase,seed=0):
 import sys,os,time,json,traceback,uuid
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path.insert(0,'/work')
 from _full_l141 import materialize,full_run,sha
 from _parity_l141 import check
 root=Path('/evidence');marker=root/(phase+'-started.json');assert not marker.exists()
 marker.write_text(json.dumps(dict(uuid=str(uuid.uuid4()),phase=phase,seed=seed,trainer_sha256=sha('/work/_full_l141.py'),model_sha256=sha('/work/relkit/relgnn_l141.py'))));volume.commit();start=time.perf_counter()
 try:
  if phase.startswith('prepare'):
   result=materialize(root/'prepared','/work/sources/l141');(root/'operator-parity.json').write_text(json.dumps(check('/work/sources/l141')));volume.commit();return result
  return full_run(root/phase,root/'prepared','/work/sources/l141',seed,phase.startswith('replay'))
 except Exception:
  (root/(phase+'-failure.txt')).write_text(traceback.format_exc());raise
 finally:
  (root/(phase+'-cost.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-start,worker_body_usd=(time.perf_counter()-start)*RATE)));volume.commit()

@app.local_entrypoint()
def main(phase:str='prepare'):
 import json
 if phase=='remaining':
  pilot=json.loads((ROOT/'labs/evidence/l141/seed-0/result.json').read_text())
  assert pilot['status']=='COMPLETE' and pilot['seconds']*1.25+120<1800
  reserve(phase,4);calls=[worker.spawn(f'seed-{s}',s) for s in range(1,5)]
  for c in calls:print(c.get())
 else:
  assert phase in ['prepare','replay','seed-0'];reserve(phase);print(worker.remote(phase,0))

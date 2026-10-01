"""Immutable USD10 reservations for the approved classification portfolio."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l151-classification-portfolio')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
for p in ['_full_l151.py','_audit_labels_l151.py','relkit/rdl_l117.py','relkit/trial_l139.py','relkit/portfolio_l151.py']:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
image=image.add_local_dir(ROOT/'labs/sources/l151','/work/sources/l151')
volume=modal.Volume.from_name('l151-portfolio-evidence',create_if_missing=True)
RATE=.00033228

def reserve(phase,workers=1,seconds=900):
 import json,fcntl,hashlib,datetime
 with (ROOT/'labs/_budget_l151.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  assert not any(x['phase']==phase for x in b['reservations']),'Duplicate reservation'
  upper=workers*seconds*RATE
  assert sum(x['upper_usd'] for x in b['reservations'])+upper+b['overhead_reserve_usd']<=b['budget_usd'],'Aggregate USD10 cap'
  b['reservations'].append(dict(phase=phase,workers=workers,timeout=seconds,upper_usd=upper,rate=RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={p:hashlib.sha256((ROOT/'labs'/p).read_bytes()).hexdigest() for p in ['_full_l151.py','_audit_labels_l151.py','relkit/rdl_l117.py','relkit/portfolio_l151.py']}))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=900,retries=0,volumes={'/evidence':volume})
def worker(phase,seed=0,lr=.0001,track='reference'):
 import sys,os,time,json,traceback,uuid
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path.insert(0,'/work')
 from _full_l151 import materialize,full_run,sha
 root=Path('/evidence');marker=root/(phase+'-started.json');assert not marker.exists()
 marker.write_text(json.dumps(dict(uuid=str(uuid.uuid4()),phase=phase,seed=seed,lr=lr,track=track,trainer_sha256=sha('/work/_full_l151.py'),model_sha256=sha('/work/relkit/rdl_l117.py'))));volume.commit();start=time.perf_counter()
 try:
  if phase=='prepare':
   from _audit_labels_l151 import audit_labels
   materialize(root/'prepared');result=audit_labels(root/'prepared');return result
  return full_run(root/phase,seed=seed,prepared_root=root/'prepared',lr=lr,track=track)
 except Exception:
  (root/(phase+'-failure.txt')).write_text(traceback.format_exc());raise
 finally:
  elapsed=time.perf_counter()-start;(root/(phase+'-cost.json')).write_text(json.dumps(dict(seconds=elapsed,worker_body_usd=elapsed*RATE)));volume.commit()

@app.local_entrypoint()
def main(phase:str='prepare'):
 import json,hashlib
 if phase in ['reference','search','selected']:
  pilot=json.loads((ROOT/'labs/evidence/l151/ref-0/result.json').read_text())
  assert pilot['status']=='COMPLETE' and pilot['seconds']*1.25+120<900
  assert json.loads((ROOT/'labs/evidence/l151/prepared/task_audit.json').read_text())['status']=='PASS'
  if phase=='reference':jobs=[(f'ref-{s}',s,.0001,'reference') for s in range(1,5)]
  elif phase=='search':jobs=[(f'search-{int(lr*1000000):03d}',100,lr,'search') for lr in [.00005,.0001,.0002]]
  else:
   freeze=json.loads((ROOT/'labs/evidence/l151/frozen.json').read_text())
   for path,h in freeze['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
   jobs=[(f'selected-{s}',s,freeze['lr'],'selected') for s in range(10,15)]
  reserve(phase,len(jobs));calls=[worker.spawn(*j) for j in jobs]
  for c in calls:print(c.get())
 else:
  assert phase in ['prepare','ref-0'];seconds=1800 if phase=='prepare' else 900
  reserve(phase,seconds=seconds);print(worker.with_options(timeout=seconds).remote(phase))

"""One bounded scale worker: USD5 maximum reservation, no automatic retries."""
import json,hashlib,fcntl,datetime
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l134-scale')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt')
 .pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html')
 .add_local_file(ROOT/'labs/_run_scale_l134.py','/work/_run_scale_l134.py')
 .add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py')
 .add_local_file(ROOT/'labs/relkit/scale_l134.py','/work/relkit/scale_l134.py')
 .add_local_file(ROOT/'labs/relkit/batch_audit_l123.py','/work/relkit/batch_audit_l123.py'))
volume=modal.Volume.from_name('l134-scale-evidence',create_if_missing=True)
RATE=.000164+2*.0000131+32*.00000222
TIMEOUT=5400
@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=TIMEOUT,retries=0,volumes={'/evidence':volume})
def worker(attempt):
 import sys,time
 sys.path.insert(0,'/work');from _run_scale_l134 import run
 root=Path('/evidence')/attempt;root.mkdir(parents=True,exist_ok=True)
 # Persistent dispatch claim prevents infrastructure restarts from silently repeating work.
 claim=root/'started.json'
 if claim.exists():raise RuntimeError('Attempt already started; fresh authorized reservation needed')
 claim.write_text(json.dumps(dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())));volume.commit()
 start=time.perf_counter()
 try:
  result=run(root);return {'status':result['status'],'rows':sum(result['rows'].values())}
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,message=str(e))));raise
 finally:
  (root/'cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,rate=RATE,resource_usd=(time.perf_counter()-start)*RATE)));volume.commit()
@app.local_entrypoint()
def main(attempt:str='attempt-1'):
 path=ROOT/'labs/_budget_l134.json'
 with path.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);entries=b.setdefault('scale_reservations',[])
  assert not any(r['attempt']==attempt for r in entries)
  assert sum(r['upper_bound_usd'] for r in entries)+TIMEOUT*RATE<=5
  assert b['maximum_worker_usd']+5+b['overhead_reserve_usd']<=10
  entries.append(dict(attempt=attempt,upper_bound_usd=TIMEOUT*RATE,timeout_s=TIMEOUT,rate=RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256((ROOT/'labs/_run_scale_l134.py').read_bytes()).hexdigest()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 print(worker.remote(attempt))

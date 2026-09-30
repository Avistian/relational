"""Immutable paid attempts; reserve every allocation before launch."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l144-contextgnn')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('optuna==4.0.0','pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
for p in ['_audit_l144.py','_full_l144.py','relkit/context_l144.py','relkit/contextgnn_l144.py']:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
image=image.add_local_dir(ROOT/'labs/sources/l144','/work/sources/l144')
volume=modal.Volume.from_name('l144-contextgnn-evidence',create_if_missing=True)
RATE=.00026124

def reserve(phase,seconds):
 import json,fcntl,hashlib,datetime
 with (ROOT/'labs/_budget_l144.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);assert not any(x['phase']==phase for x in b['reservations'])
  upper=seconds*RATE;assert sum(x['upper_usd'] for x in b['reservations'])+upper+b['overhead_reserve_usd']<=b['budget_usd'],'Aggregate cap'
  b['reservations'].append(dict(phase=phase,timeout=seconds,upper_usd=upper,rate_per_second=RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={p:hashlib.sha256((ROOT/'labs'/p).read_bytes()).hexdigest() for p in ['_audit_l144.py','_full_l144.py','relkit/context_l144.py','relkit/contextgnn_l144.py']}));f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=1800,retries=0,volumes={'/evidence':volume})
def worker(phase):
 import os,sys,time,json,traceback
 os.environ.update(XDG_CACHE_HOME='/evidence/cache',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path[:0]=['/work','/work/sources/l144']
 from _full_l144 import prepare,run_fit,PILOT_CONFIG
 root=Path('/evidence');marker=root/(phase+'-started.json');assert not marker.exists();marker.write_text(json.dumps({'phase':phase}));volume.commit();start=time.perf_counter()
 try:
  if phase.startswith('prepare'):return prepare(root/'prepared','/work/sources/l144')
  if not (root/'independent-labels.json').exists():
   from _audit_l144 import independent_labels
   independent_labels(root/'independent-labels.json');volume.commit()
  arm='contextgnn' if 'contextgnn' in phase else 'shallowrhsgnn'
  return run_fit(root/'prepared',root/phase,arm,PILOT_CONFIG,epochs=1,pilot=True)
 except Exception:
  (root/(phase+'-failure.txt')).write_text(traceback.format_exc());raise
 finally:
  (root/(phase+'-cost.json')).write_text(json.dumps({'seconds':time.perf_counter()-start,'worker_body_usd':(time.perf_counter()-start)*RATE}));volume.commit()

@app.local_entrypoint()
def main(phase:str='prepare'):
 assert phase in ['prepare','pilot-contextgnn','pilot-shallowrhsgnn','prepare-retry1','pilot-contextgnn-retry1','pilot-shallowrhsgnn-retry1']
 reserve(phase,1800);print(worker.remote(phase))

@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=10800,retries=0,volumes={'/evidence':volume})
def search_worker(arm):
 import sys,os,json,time,traceback
 os.environ.update(XDG_CACHE_HOME='/evidence/cache',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path[:0]=['/work','/work/sources/l144']
 from _full_l144 import full_search
 start=time.perf_counter();root=Path('/evidence');phase='search-'+arm
 assert not (root/(phase+'-started.json')).exists();(root/(phase+'-started.json')).write_text(json.dumps({'arm':arm}));volume.commit()
 try:return full_search(root/'prepared',root/phase,arm)
 except Exception:
  (root/(phase+'-failure.txt')).write_text(traceback.format_exc());raise
 finally:
  (root/(phase+'-cost.json')).write_text(json.dumps({'seconds':time.perf_counter()-start,'worker_body_usd':(time.perf_counter()-start)*RATE}));volume.commit()

@app.local_entrypoint(name='full')
def full():
 import json
 estimate=json.loads((ROOT/'labs/evidence/l144/cost-decision.json').read_text())
 assert estimate['decision']=='PROCEED','Pilot projection blocks full dispatch within USD10'
 # Reserve both arms before either starts; three-hour allocation limit per arm.
 reserve('search-contextgnn',10800);reserve('search-shallowrhsgnn',10800)
 for arm in ['contextgnn','shallowrhsgnn']:print(search_worker.remote(arm))

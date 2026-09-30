"""USD10 aggregate: reserve immutable attempts before launching any worker."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l145-relgt');RATE=.00022572
image=(modal.Image.debian_slim(python_version='3.11').env({'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}).pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('h5py==3.12.1','einops==0.8.0','pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
for p in ['_full_l145.py','_mechanism_l145.py','relkit/relgt_l145.py','relkit/relgt_contracts_l145.py']:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
image=image.add_local_dir(ROOT/'labs/sources/l145','/work/sources/l145')
v=modal.Volume.from_name('l145-relgt-evidence',create_if_missing=True);prior=modal.Volume.from_name('l143-relgnn-evidence')

def reserve(phase,seconds):
 import json,fcntl,datetime,hashlib
 with (ROOT/'labs/_budget_l145.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);assert not any(r['phase']==phase for r in b['reservations'])
  upper=seconds*RATE;assert sum(r['upper_usd'] for r in b['reservations'])+upper+b['overhead_reserve_usd']<=b['budget_usd']
  b['reservations'].append(dict(phase=phase,timeout=seconds,upper_usd=upper,rate=RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hashes={p:hashlib.sha256((ROOT/'labs'/p).read_bytes()).hexdigest() for p in ['_full_l145.py','relkit/relgt_l145.py','relkit/relgt_contracts_l145.py']}));f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1200,retries=0,volumes={'/evidence':v,'/prior':prior.read_only()})
def worker(phase):
 import sys,time,json,traceback
 sys.path.insert(0,'/work');from _full_l145 import prepare,run_fit
 start=time.perf_counter();root=Path('/evidence');assert not (root/(phase+'-started.json')).exists()
 (root/(phase+'-started.json')).write_text(json.dumps({'phase':phase}));v.commit()
 try:
  if phase.startswith('prepare'):
   from _mechanism_l145 import mechanism_check
   (root/'mechanism-pinned.json').write_text(json.dumps(mechanism_check('/work/sources/l145')))
   return prepare(root/'prepared','/prior/prepared','/work/sources/l145')
  return run_fit(root/'prepared','/prior/prepared','/work/sources/l145',root/phase,layers=int(phase.split('-')[1]),pilot=True)
 except Exception:
  (root/(phase+'-failure.txt')).write_text(traceback.format_exc());raise
 finally:
  seconds=time.perf_counter()-start;(root/(phase+'-cost.json')).write_text(json.dumps({'seconds':seconds,'worker_body_usd':seconds*RATE}));v.commit()

@app.local_entrypoint()
def main(phase:str='prepare'):
 reserve(phase,1200);print(worker.remote(phase))

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=18000,retries=0,volumes={'/evidence':v,'/prior':prior.read_only()})
def complete_worker():
 import sys,time,json,traceback
 sys.path.insert(0,'/work');from _full_l145 import full_search
 start=time.perf_counter()
 try:return full_search('/evidence/prepared','/prior/prepared','/work/sources/l145','/evidence/full')
 finally:
  seconds=time.perf_counter()-start;Path('/evidence/full-cost.json').write_text(json.dumps({'seconds':seconds,'worker_body_usd':seconds*RATE}));v.commit()

@app.local_entrypoint(name='full')
def full():
 import json
 audit=json.loads((ROOT/'labs/evidence/l145/prepared/audit.json').read_text())
 decision=json.loads((ROOT/'labs/evidence/l145/cost-decision.json').read_text())
 assert audit['temporal_status']=='PASS','Temporal contract failed; no clean reproduction dispatch'
 assert decision['decision']=='PROCEED' and decision['projected_worker_seconds']<18000,'Full cost projection must fit reserved runtime'
 reserve('full',18000);print(complete_worker.remote())

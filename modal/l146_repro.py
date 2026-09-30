"""Reserve aggregate cost before immutable, bounded L146 dispatches."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l146-comparison');RATE=.00022572
image=(modal.Image.debian_slim(python_version='3.11').env({'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}).pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('h5py==3.12.1','einops==0.8.0','pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
FILES=['_full_l146.py','_full_l145.py','relkit/comparison_l146.py','relkit/gnn_l146.py','relkit/relgt_course_l146.py','relkit/relgt_l145.py','relkit/relgt_contracts_l145.py']
for p in FILES:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
v=modal.Volume.from_name('l146-comparison-evidence',create_if_missing=True);prior=modal.Volume.from_name('l143-relgnn-evidence')

def reserve(phase,seconds):
 import json,fcntl,datetime,hashlib
 with (ROOT/'labs/_budget_l146.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);assert not any(r['phase']==phase for r in b['reservations'])
  upper=seconds*RATE;assert sum(r['upper_usd'] for r in b['reservations'])+upper+b['overhead_reserve_usd']<=b['budget_usd']
  b['reservations'].append(dict(phase=phase,timeout=seconds,upper_usd=upper,rate=RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hashes={p:hashlib.sha256((ROOT/'labs'/p).read_bytes()).hexdigest() for p in FILES}));f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0,volumes={'/evidence':v,'/prior':prior.read_only()})
def worker(phase):
 import sys,time,json,traceback
 sys.path.insert(0,'/work');from _full_l146 import prepare,fit,recover_completed_fits
 start=time.perf_counter();root=Path('/evidence');assert not (root/(phase+'-started.json')).exists()
 (root/(phase+'-started.json')).write_text(json.dumps({'phase':phase}));v.commit()
 try:
  if phase=='prepare':return prepare(root/'prepared','/prior/prepared')
  if phase=='recover':return recover_completed_fits(root,'/prior/prepared')
  mode,arm,seed=phase.split('-');return fit(root/'prepared','/prior/prepared',root/phase,arm,int(seed),pilot=mode=='pilot')
 except Exception:
  (root/(phase+'-failure.txt')).write_text(traceback.format_exc());raise
 finally:
  seconds=time.perf_counter()-start;(root/(phase+'-cost.json')).write_text(json.dumps({'seconds':seconds,'worker_body_usd':seconds*RATE}));v.commit()

@app.local_entrypoint()
def main(phase:str='prepare'):
 import json
 if phase.startswith('fit-'):
  decision=json.loads((ROOT/'labs/evidence/l146/cost-decision.json').read_text())
  assert decision['decision']=='PROCEED' and decision['projected_complete_worker_usd']<6
 reserve(phase,1800);print(worker.remote(phase))

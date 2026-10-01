"""Approved USD10 aggregate gate: every immutable allocation reserved before launch."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];RATE=.00026124;app=modal.App('l153-recommendation-portfolio')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
FILES=['_prepare_l153.py','_run_l153.py','_labels_l153.py','relkit/recommendation_model_l153.py','relkit/recommendation_l153.py','relkit/batch_audit_l123.py']
for name in FILES:image=image.add_local_file(ROOT/'labs'/name,'/work/'+name)
image=image.add_local_dir(ROOT/'labs/sources/l153','/work/sources/l153')
volume=modal.Volume.from_name('l153-recommendation-evidence',create_if_missing=True)

def reserve(phase,seconds,workers=1):
 import json,fcntl,hashlib,datetime
 with (ROOT/'labs/_budget_l153.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  assert not any(r['phase']==phase for r in b['reservations']),'Attempt already reserved'
  for p,h in b['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
  upper=workers*seconds*RATE
  assert upper+sum(r['upper_usd'] for r in b['reservations'])+b['overhead_reserve_usd']<=b['budget_usd'],'USD10 aggregate cap'
  b['reservations'].append(dict(phase=phase,workers=workers,seconds=seconds,upper_usd=upper,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()

def execute(phase,seed):
 import os,sys,time,json,traceback
 os.environ.update(XDG_CACHE_HOME='/evidence/cache',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path[:0]=['/work','/work/sources/l153']
 import torch
 torch.set_num_threads(1)
 from _prepare_l153 import prepare
 from _run_l153 import run_fit
 root=Path('/evidence');marker=root/(phase+'-started.json');assert not marker.exists();marker.write_text(json.dumps(dict(phase=phase,seed=seed)));volume.commit();start=time.perf_counter()
 try:
  r=prepare(root/'prepared','/work/sources/l153') if phase.startswith('prepare') else run_fit(root/'prepared',root/phase,seed,pilot=phase.startswith('pilot'))
  (root/(phase+'-completed.json')).write_text(json.dumps(dict(phase=phase,status='COMPLETE',seconds=time.perf_counter()-start)));return {k:v for k,v in r.items() if k not in ['database_files','task_files','rows','stypes','trace']}
 except Exception:
  (root/(phase+'-failure.txt')).write_text(traceback.format_exc());raise
 finally:
  elapsed=time.perf_counter()-start;(root/(phase+'-cost.json')).write_text(json.dumps(dict(seconds=elapsed,worker_body_usd=elapsed*RATE)));volume.commit()

@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=1800,retries=0,volumes={'/evidence':volume})
def probe(phase):return execute(phase,0)

@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=4200,retries=0,volumes={'/evidence':volume})
def fit(seed):return execute('seed-'+str(seed),seed)

@app.local_entrypoint()
def main(phase:str='prepare'):
 import json
 if phase in ['prepare','pilot']:
  reserve(phase,1800);print(probe.remote(phase))
 elif phase=='full':
  decision=json.loads((ROOT/'labs/evidence/l153/cost-decision.json').read_text());assert decision['decision']=='PROCEED','Full five-seed projection exceeds cap'
  reserve('full-five-seeds',4200,5)
  for seed in range(5):print(fit.remote(seed))
 else:raise ValueError(phase)

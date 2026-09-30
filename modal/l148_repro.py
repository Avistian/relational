"""Approved25fit experiment; reserve each call before dispatch."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];RATE=.00022572
app=modal.App('l148-ablation-discipline')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html').add_local_dir(ROOT/'labs/relkit','/work/relkit').add_local_dir(ROOT/'labs/sources/l117','/work/sources/l117'))
for name in ['_train_l148.py','_prepare_l148.py','_neural_l148.py','_real_probe_l148.py']:image=image.add_local_file(ROOT/'labs'/name,'/work/'+name)
volume=modal.Volume.from_name('l148-ablation-evidence',create_if_missing=True)
cache=modal.Volume.from_name('l148-ablation-cache',create_if_missing=True)

def reserve(phases,seconds):
 import json,fcntl,datetime,hashlib
 p=ROOT/'labs/_budget_l148.json'
 with p.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  assert not set(phases)&{r['phase'] for r in b['reservations']},'Duplicate dispatch'
  assert sum(r['upper_usd'] for r in b['reservations'])+len(phases)*seconds*RATE+3<=10
  for phase in phases:b['reservations'].append(dict(phase=phase,seconds=seconds,upper_usd=seconds*RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [ROOT/'labs/_train_l148.py',ROOT/'labs/relkit/ablation_model_l148.py',ROOT/'labs/relkit/ablation_l148.py']}))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=1800,retries=0,volumes={'/evidence':volume,'/cache':cache})
def prepare():
 import sys,time,json,os
 os.environ.update(XDG_CACHE_HOME='/cache',HF_HOME='/cache/huggingface')
 sys.path.insert(0,'/work');from _prepare_l148 import prepare
 out=Path('/evidence/prepared');out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 try:return prepare(out)
 finally:
  (out/'cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,worker_body_usd=(time.perf_counter()-start)*RATE)));volume.commit();cache.commit()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=900,retries=0,volumes={'/evidence':volume,'/cache':cache})
def fit(arm,seed):
 import sys,time,json,torch,uuid,hashlib,traceback,os
 os.environ.update(XDG_CACHE_HOME='/cache',HF_HOME='/cache/huggingface')
 sys.path.insert(0,'/work');from _train_l148 import fit_ablation
 torch.set_num_threads(1);out=Path('/evidence')/f'{arm}-{seed}';out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 run_id=str(uuid.uuid4());(out/'started.json').write_text(json.dumps(dict(run_id=run_id,arm=arm,seed=seed)))
 try:
  prepared=Path('/evidence/prepared');audit=json.loads((prepared/'audit.json').read_text());assert hashlib.sha256((prepared/'graph.pt').read_bytes()).hexdigest()==audit['graph_sha256']
  g=torch.load(prepared/'graph.pt',weights_only=False)
  sys.path.insert(0,'/work/sources/l117');from model import Model
  r=fit_ablation(g['data'],g['stats'],g['task'],seed,out,10,'cuda',Model if arm=='full' else None,arm)
  r['run_id']=run_id;r['graph_sha256']=audit['graph_sha256'];(out/'result.json').write_text(json.dumps(r,indent=2));return dict(arm=arm,seed=seed,scores=r['scores'],seconds=r['seconds'])
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  elapsed=time.perf_counter()-start;(out/'cost.json').write_text(json.dumps(dict(seconds=elapsed,worker_body_usd=elapsed*RATE)));volume.commit()

@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=600,retries=0,volumes={'/evidence':volume,'/cache':cache})
def audit():
 import sys,os,json,time
 os.environ.update(XDG_CACHE_HOME='/cache',HF_HOME='/cache/huggingface');sys.path.insert(0,'/work')
 from _neural_l148 import run as neural
 from _real_probe_l148 import run as real
 start=time.perf_counter()
 try:
  r=neural();Path('/evidence/neural-pinned.json').write_text(json.dumps(r,indent=2));return real('/evidence')
 finally:
  Path('/evidence/audit-cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,worker_body_usd=(time.perf_counter()-start)*RATE)));volume.commit()

@app.local_entrypoint()
def main(phase:str='prepare'):
 import json
 if phase in ['prepare','prepare-recovery','prepare-runtime-cache']:reserve([phase],1800);print(prepare.remote())
 elif phase=='audit':reserve(['audit'],600);print(audit.remote())
 elif phase=='pilot':reserve(['full-0'],900);print(fit.remote('full',0))
 elif phase=='remaining':
  gate=json.loads((ROOT/'labs/evidence/l148/pilot-gate.json').read_text());assert gate['approved']
  pairs=[(a,s) for a in ['full','encoder','messages','history','combined'] for s in range(5) if (a,s)!=('full',0)]
  reserve([f'{a}-{s}' for a,s in pairs],900)
  for result in fit.starmap(pairs):print(result)
 else:raise ValueError(phase)

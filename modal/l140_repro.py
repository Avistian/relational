"""Full two-task checkpoint. Reservations persist; no automatic retries."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l140-two-task-checkpoint')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
image=image.add_local_file(ROOT/'labs/_full_l140.py','/work/_full_l140.py').add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py').add_local_dir(ROOT/'labs/sources/l140','/work/sources/l140')
volume=modal.Volume.from_name('l140-checkpoint-evidence',create_if_missing=True)
prior={k:modal.Volume.from_name(v) for k,v in [('amazon','l138-amazon-evidence'),('trial','l139-trial-evidence')]}
volumes={'/evidence':volume,**{'/prior/'+k:v for k,v in prior.items()}}
RATE=.000164+2*.0000131+64*.00000222
TIMEOUTS={'amazon':2300,'trial':600}

def reserve(phase, seconds, rate, workers=1):
 import json,hashlib,fcntl,datetime
 with (ROOT/'labs/_budget_l140.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  assert not any(r['phase']==phase for r in b['reservations']),'Attempt already reserved'
  upper=seconds*rate*workers
  assert sum(r['upper_usd'] for r in b['reservations'])+upper+b['overhead_reserve_usd']<=b['budget_usd'],'USD10 aggregate cap'
  b['reservations'].append(dict(phase=phase,workers=workers,timeout=seconds,rate=rate,upper_usd=upper,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'labs/_full_l140.py',ROOT/'labs/relkit/rdl_l117.py',ROOT/'modal/l140_repro.py']}))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()

@app.function(image=image,cpu=2,memory=8192,timeout=600,retries=0,volumes=volumes)
def preflight():
 import sys,json,time,hashlib
 sys.path.insert(0,'/work');from _full_l140 import TASKS,sha
 out=Path('/evidence');assert not (out/'preflight-started').exists();(out/'preflight-started').touch();volume.commit();start=time.perf_counter();report={}
 try:
  for name,cfg in TASKS.items():
   root=Path('/prior')/name;archives={}
   for file,digest in cfg['archives'].items():
    actual=sha(root/'cache'/file);assert actual==digest;archives[file]=actual
   provenance=json.loads((root/('training_pilot.json' if name=='amazon' else 'prepared.json')).read_text())
   digest=sha(root/'graph.pt')
   if name=='trial':assert digest==provenance['graph_sha256']
   report[name]=dict(graph_sha256=digest,graph_bytes=(root/'graph.pt').stat().st_size,archives=archives,preprocessing='REUSED; current hash checked, historical graph identity not established',provenance=provenance)
  (out/'preflight.json').write_text(json.dumps(report,indent=2));return report
 finally:
  (out/'preflight-cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,resource_usd=(time.perf_counter()-start)*(2*.0000131+8*.00000222))));volume.commit()

@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=2300,retries=0,volumes=volumes)
def fit(task_name,seed):
 import sys,json,time,traceback,os,uuid,hashlib
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.path.insert(0,'/work')
 from _full_l140 import full_run
 root=Path('/evidence');name=f'{task_name}/seed-{seed}';marker=root/(name.replace('/','-')+'-started.json');assert not marker.exists(),'No automatic redispatch'
 identity=dict(run_uuid=str(uuid.uuid4()),started=time.time(),task=task_name,seed=seed,source_sha256=hashlib.sha256(Path('/work/_full_l140.py').read_bytes()).hexdigest())
 marker.write_text(json.dumps(identity));volume.commit();start=time.perf_counter()
 try:
  result=full_run(root/name,task_name,seed,Path('/prior')/task_name)
  (root/name/'run_identity.json').write_text(json.dumps(identity,indent=2));return {'task':task_name,'seed':seed,'seconds':result['seconds'],'scores':result['scores']}
 except Exception:
  (root/(name.replace('/','-')+'-error.txt')).write_text(traceback.format_exc());raise
 finally:
  (root/(name.replace('/','-')+'-cost.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-start,resource_usd=(time.perf_counter()-start)*RATE)));volume.commit()

@app.local_entrypoint()
def main(phase:str='preflight'):
 import json
 if phase=='preflight':
  reserve(phase,600,2*.0000131+8*.00000222);print(preflight.remote());return
 assert phase in ['pilots','remaining']
 seeds=[0] if phase=='pilots' else list(range(1,5))
 # Complete seed0 is the timing pilot and is retained in the final five.
 if phase=='remaining':
  for name in TIMEOUTS:
   d=json.loads((ROOT/f'labs/evidence/l140/{name}/seed-0/result.json').read_text())
   assert d['status']=='COMPLETE' and d['seconds']*1.25+120<TIMEOUTS[name],('Forecast exceeds worker cutoff',name)
 for name in TIMEOUTS:reserve(phase+'-'+name,TIMEOUTS[name],RATE,len(seeds))
 calls=[]
 for name in TIMEOUTS:
  fn=fit.with_options(timeout=TIMEOUTS[name])
  for seed in seeds:calls.append(fn.spawn(name,seed))
 for call in calls:print(call.get())

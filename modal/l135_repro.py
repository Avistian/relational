"""Budgeted L135 pilot, validation-only search, frozen final comparison."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l135-budgeted-tuning')
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt')
 .pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
for path in ['relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/tuning_l135.py','relkit/tuning_train_l135.py','_run_l117.py','_run_l135.py','_protocol_l135.json']:
 image=image.add_local_file(ROOT/'labs'/path,'/work/'+path)
image=image.add_local_dir(ROOT/'labs/sources/l117','/work/sources/l117')
volume=modal.Volume.from_name('l135-tuning-evidence',create_if_missing=True)
RATE=.00022572
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=900,retries=0,volumes={'/evidence':volume})
def worker(seed,epochs,configuration,phase,test):
 import sys,time,json,datetime,hashlib,traceback
 sys.path.insert(0,'/work');from _run_l135 import run
 root=Path('/evidence')/phase/configuration['id']/f'seed-{seed}'
 assert not root.exists(),'Refuse to overwrite any prior attempt'
 root.mkdir(parents=True);start=time.perf_counter()
 try:
  result=run(seed,epochs,root,configuration,test)
  done=dict(status='COMPLETE',seed=seed,config=configuration['id'],epochs=epochs,phase=phase,selection_mae=result['selection_mae'],scores=result['scores'],seconds=time.perf_counter()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  done['resource_usd']=done['seconds']*RATE
  done['source_hashes']={p:hashlib.sha256(Path('/work',p).read_bytes()).hexdigest() for p in ['relkit/tuning_train_l135.py','relkit/rdl_l117.py','_run_l135.py','_protocol_l135.json']}
  (root/'completed.json').write_text(json.dumps(done,indent=2));return done
 except Exception:
  (root/'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start),indent=2));raise
 finally:volume.commit()

@app.local_entrypoint()
def main(mode:str='pilot'):
 import sys,json,hashlib,fcntl,datetime
 sys.path.insert(0,str(ROOT/'labs'));from relkit.tuning_l135 import reserve_budget
 protocol=json.loads((ROOT/'labs/_protocol_l135.json').read_text());configs={c['id']:c for c in protocol['configurations']}
 assert mode in ['pilot','search','final']
 if mode=='pilot':jobs=[(999,1,configs[protocol['default']],'pilot',False)]
 elif mode=='search':jobs=[(s,10,c,'search',False) for c in configs.values() for s in protocol['search_seeds']]
 else:
  freeze=json.loads((ROOT/'labs/evidence/l135/frozen.json').read_text())
  assert freeze['protocol_sha256']==hashlib.sha256((ROOT/'labs/_protocol_l135.json').read_bytes()).hexdigest()
  # Freeze includes exact bytes of every search result; later changes are rejected.
  for p,sha in freeze['search_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha
  arms=list(dict.fromkeys([protocol['default'],freeze['winner']]))
  jobs=[(s,10,configs[c],'final',True) for c in arms for s in protocol['final_seeds']]
 with (ROOT/'labs/_budget_l135.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  for p,sha in b['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p
  assert not any(r['phase']==mode for r in b['reservations'])
  if mode!='pilot':assert b['pilot_passed']
  total=reserve_budget([r['upper_usd'] for r in b['reservations']],len(jobs),900,RATE,3,10)
  entry=dict(phase=mode,workers=len(jobs),upper_usd=len(jobs)*900*RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  if mode=='final':entry['freeze_sha256']=hashlib.sha256((ROOT/'labs/evidence/l135/frozen.json').read_bytes()).hexdigest()
  b['reservations'].append(entry);b['total_reserved_usd']=total
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 for done in worker.starmap(jobs):print(done)

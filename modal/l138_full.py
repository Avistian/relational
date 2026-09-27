"""Five complete released-protocol fits; refuses unaffordable dispatch."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l138-full-reproduction')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
image=image.add_local_file(ROOT/'labs/_full_l138.py','/work/_full_l138.py').add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py')
volume=modal.Volume.from_name('l138-amazon-evidence');RATE=.00033228
@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=3000,retries=0,volumes={'/evidence':volume})
def fit(seed):
 import sys,time,json,traceback
 sys.path.insert(0,'/work');from _full_l138 import full_run
 start=time.perf_counter();dest=Path('/evidence/final')/f'seed-{seed}'
 try:return full_run(dest,seed,10,prepared_root='/evidence')
 except Exception:
  dest.mkdir(parents=True,exist_ok=True);(dest/'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start),indent=2));raise
 finally:volume.commit()
@app.local_entrypoint()
def main():
 import json,fcntl,hashlib
 pilot=json.loads((ROOT/'labs/evidence/l138/training_pilot.json').read_text())
 assert pilot['status']=='COMPLETE','Complete full-graph timing first'
 assert pilot['projected_five_run_seconds']/5*1.25+120<=3000,'Five-run forecast exceeds reserved worker runtime; no dispatch'
 with (ROOT/'labs/_budget_l138.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);upper=5*3000*RATE
  assert not any(x['phase']=='full_five_seeds' for x in b['reservations'])
  assert sum(x['upper_usd'] for x in b['reservations'])+upper+3<=10,'Aggregate cap exceeded'
  b['reservations'].append(dict(phase='full_five_seeds',upper_usd=upper,workers=5,timeout=3000,source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['labs/_full_l138.py','labs/relkit/rdl_l117.py','labs/requirements-l117-runtime.txt','modal/l138_full.py']}))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 for result in fit.map(range(5)):print(result)

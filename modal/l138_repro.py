"""L138 source-pinned workers; USD10 aggregate guard before every dispatch."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l138-amazon-reproduction')
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt')
 .pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
image=image.add_local_dir(ROOT/'labs/sources/l138','/work/sources/l138').add_local_file(ROOT/'labs/_probe_l138.py','/work/_probe_l138.py')
volume=modal.Volume.from_name('l138-amazon-evidence',create_if_missing=True)
RATE=.00026124
@app.function(image=image,gpu='T4',cpu=2,memory=32768,timeout=1800,retries=0,volumes={'/evidence':volume})
def probe():
 import sys,time,json,traceback
 sys.path.insert(0,'/work');from _probe_l138 import run
 root=Path('/evidence');start=time.perf_counter()
 try:return run(root)
 except Exception:
  (root/'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start),indent=2));raise
 finally:volume.commit()
@app.local_entrypoint()
def main():
 import json,fcntl,hashlib,datetime
 budget=ROOT/'labs/_budget_l138.json'
 with budget.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);assert not b['reservations'],'Probe already dispatched'
  upper=1800*RATE;assert upper+b['overhead_reserve_usd']<=b['budget_usd']
  b['reservations'].append(dict(phase='probe',upper_usd=upper,timeout=1800,rate=RATE,source_sha256=hashlib.sha256((ROOT/'labs/_probe_l138.py').read_bytes()).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 print(probe.remote())

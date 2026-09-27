"""Full graph pilot with explicit worst-case reservation; no automatic retry."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l138-full-graph-pilot')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
image=image.add_local_file(ROOT/'labs/_pilot_l138.py','/work/_pilot_l138.py').add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py')
volume=modal.Volume.from_name('l138-amazon-evidence');RATE=.000164+2*.0000131+128*.00000222
@app.function(image=image,gpu='T4',cpu=2,memory=131072,timeout=2700,retries=0,volumes={'/evidence':volume})
def pilot():
 import sys,time,json,traceback
 sys.path.insert(0,'/work');from _pilot_l138 import run
 start=time.perf_counter()
 try:return run('/evidence')
 except Exception:
  Path('/evidence/pilot_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start),indent=2));raise
 finally:volume.commit()
@app.local_entrypoint()
def main():
 import json,fcntl,hashlib
 p=ROOT/'labs/_budget_l138.json'
 with p.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);upper=2700*RATE
  assert not any(x['phase']=='full_graph_pilot' for x in b['reservations'])
  assert sum(x['upper_usd'] for x in b['reservations'])+upper+3<=10
  b['reservations'].append(dict(phase='full_graph_pilot',upper_usd=upper,timeout=2700,rate=RATE,source_sha256=hashlib.sha256((ROOT/'labs/_pilot_l138.py').read_bytes()).hexdigest()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 print(pilot.remote())

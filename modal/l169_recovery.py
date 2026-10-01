"""One explicitly reserved restart after client deadline; original inference code unchanged."""
import json,sys,time,traceback
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'labs'))
app=modal.App('l169-context-scaling-recovery')
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install('numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','pydantic==1.10.26','pyyaml==6.0.2','tabicl==0.1.3')
 .add_local_dir(ROOT/'labs/sources/l166/upstream/model_pretrain','/source/model_pretrain')
 .add_local_file(ROOT/'labs/_run_l169.py','/work/_run_l169.py').add_local_dir('/tmp/l169-input','/input'))
volume=modal.Volume.from_name('l169-context-scaling-evidence')
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=1200,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def retry_remaining():
 import sys
 sys.path.insert(0,'/work')
 from _run_l169 import run169
 out=Path('/evidence/remaining-2');assert not out.exists();out.mkdir();start=time.perf_counter()
 try:return run169('/input','/source',out,'remaining')
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  (out/'cost.json').write_text(json.dumps(dict(worker_body_seconds=time.perf_counter()-start,rate=.00028372)));volume.commit()
@app.local_entrypoint()
def main():
 from _guard_l166 import reserve
 path=ROOT/'labs/evidence/l169/budget.json';b=json.loads(path.read_text())
 assert sum(r['seconds'] for r in b['reservations'])+1200<=21600
 reserve(path,ROOT,'remaining-2',1200)
 print('runs',len(retry_remaining.remote()['records']))

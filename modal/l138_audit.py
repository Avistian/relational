"""Full-label audit, charged to the L138 aggregate budget."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l138-amazon-task-audit')
image=modal.Image.debian_slim(python_version='3.11').pip_install('numpy==1.26.4','pandas==2.2.3','pyarrow==18.1.0','scikit-learn==1.5.2').add_local_file(ROOT/'labs/_audit_task_l138.py','/work/_audit_task_l138.py')
volume=modal.Volume.from_name('l138-amazon-evidence')
@app.function(image=image,cpu=2,memory=16384,timeout=1200,retries=0,volumes={'/evidence':volume})
def audit():
 import sys,time,json,traceback
 sys.path.insert(0,'/work');from _audit_task_l138 import run
 start=time.perf_counter()
 try:return run('/evidence')
 except Exception:
  Path('/evidence/audit_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start),indent=2));raise
 finally:volume.commit()
@app.local_entrypoint()
def main():
 import json,fcntl,hashlib
 p=ROOT/'labs/_budget_l138.json'
 with p.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);upper=1200*(2*.0000131+16*.00000222)
  assert not any(x['phase']=='task_audit' for x in b['reservations'])
  assert sum(x['upper_usd'] for x in b['reservations'])+upper+3<=10
  b['reservations'].append(dict(phase='task_audit',upper_usd=upper,timeout=1200,source_sha256=hashlib.sha256((ROOT/'labs/_audit_task_l138.py').read_bytes()).hexdigest()))
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 print(audit.remote())

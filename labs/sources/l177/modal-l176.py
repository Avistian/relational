"""Bounded nonblocking L176 inference; immutable attempts and no automatic retries."""
import json,sys,time,traceback,hashlib
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'labs'))
RATE=.00028372
app=modal.App('l176-nested-support')
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install('numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','pydantic==1.10.26','pyyaml==6.0.2','tabicl==0.1.3')
 .add_local_dir(ROOT/'labs/sources/l166/upstream/model_pretrain','/source/model_pretrain')
 .add_local_file(ROOT/'labs/_run_l176.py','/work/_run_l176.py')
 .add_local_dir('/tmp/l176-input','/input'))
volume=modal.Volume.from_name('l176-nested-support-evidence',create_if_missing=True)
def execute(phase):
 import sys
 sys.path.insert(0,'/work')
 from _run_l176 import run176
 out=Path('/evidence')/(phase+'-1');assert not out.exists(),'Immutable attempt exists'
 out.mkdir();start=time.perf_counter()
 try:return run176('/input','/source',out,phase)
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  (out/'cost.json').write_text(json.dumps(dict(worker_body_seconds=time.perf_counter()-start,rate=RATE)))
  volume.commit()
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=600,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def probe():return execute('pilot')
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=7200,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def remaining():return execute('remaining')
@app.local_entrypoint()
def main(phase:str='pilot'):
 from relkit.few_shot_l176 import reserve_cost
 E=ROOT/'labs/evidence/l176';b=json.loads((E/'budget.json').read_text())
 if phase not in ['pilot','remaining']:raise ValueError('Unknown phase')
 for name,h in b['source_files'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'Source changed: '+name
 assert json.loads((E/'preflight.json').read_text())['status']=='PASS'
 if phase=='remaining':assert json.loads((E/'cost-decision.json').read_text())['decision']=='PROCEED'
 b=reserve_cost(b,phase+'-1',600 if phase=='pilot' else 7200,RATE)
 (E/'budget.json').write_text(json.dumps(b,indent=2)+'\n')
 call=(probe if phase=='pilot' else remaining).spawn()
 (E/(phase+'-call.json')).write_text(json.dumps(dict(call_id=call.object_id))+'\n');print(call.object_id)

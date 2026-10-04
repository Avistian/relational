"""Fresh approved 30 evaluations; source runner unchanged from authenticated L166."""
import json,sys,time,traceback
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('b23-rdbpfn-comparison')
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install('numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','pydantic==1.10.26','pyyaml==6.0.2','tabicl==0.1.3')
 .add_local_dir(ROOT/'labs/sources/l166/upstream/model_pretrain','/source/model_pretrain')
 .add_local_file(ROOT/'labs/_run_l166.py','/work/_run_l166.py')
 .add_local_dir('/tmp/l166-input','/input'))
volume=modal.Volume.from_name('b23-rdbpfn-evidence',create_if_missing=True)
def execute(phase,seeds):
 sys.path.insert(0,'/work')
 from _run_l166 import run166
 out=Path('/evidence')/phase
 if out.exists():raise RuntimeError('Immutable run already exists')
 out.mkdir();start=time.monotonic()
 try:return run166('/input','/source',out,seeds)
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  (out/'cost.json').write_text(json.dumps(dict(worker_body_seconds=time.monotonic()-start,rate=.00028372)));volume.commit()
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=600,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def pilot():return execute('pilot-1',[0])
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=5300,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def remaining():return execute('full-1',list(range(1,10)))
@app.local_entrypoint()
def main(phase:str='pilot'):
 sys.path.insert(0,str(ROOT/'labs'))
 from _guard_l200 import reserve
 budget=ROOT/'labs/evidence/b23/budget.json'
 if phase=='pilot':reserve(budget,ROOT,'pilot-1',600);result=pilot.remote()
 elif phase=='full':
  d=json.loads((ROOT/'labs/evidence/b23/admission.json').read_text())
  if d['decision']!='PROCEED':raise RuntimeError('Pilot did not admit full scope')
  reserve(budget,ROOT,'full-1',5300);result=remaining.remote()
 else:raise ValueError('Unknown phase')
 (ROOT/'labs/evidence/b23'/('remote-'+phase+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

"""Approved context sweep with immutable reservations and no automatic retries."""
import json,sys,time,traceback
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'labs'))
RATE=.00028372
app=modal.App('l169-context-scaling')
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
       .pip_install('numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','pydantic==1.10.26','pyyaml==6.0.2','tabicl==0.1.3')
       .add_local_dir(ROOT/'labs/sources/l166/upstream/model_pretrain','/source/model_pretrain')
       .add_local_file(ROOT/'labs/_run_l169.py','/work/_run_l169.py')
       .add_local_dir('/tmp/l169-input','/input'))
volume=modal.Volume.from_name('l169-context-scaling-evidence',create_if_missing=True)
def execute(phase):
    import sys
    sys.path.insert(0,'/work')
    from _run_l169 import run169
    out=Path('/evidence')/(phase+'-1');assert not out.exists(),'Immutable attempt exists'
    out.mkdir();start=time.perf_counter()
    try:return run169('/input','/source',out,phase)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        (out/'cost.json').write_text(json.dumps(dict(worker_body_seconds=time.perf_counter()-start,rate=RATE)));volume.commit()
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=600,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def probe():return execute('pilot')
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=7200,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def remaining():return execute('remaining')
@app.local_entrypoint()
def main(phase:str='pilot'):
    from _guard_l166 import reserve
    if phase not in ['pilot','remaining']:raise ValueError('Unknown phase')
    path=ROOT/'labs/evidence/l169/budget.json';b=json.loads(path.read_text());seconds=600 if phase=='pilot' else 7200
    assert sum(r['seconds'] for r in b['reservations'])+seconds<=21600
    if phase=='remaining':assert json.loads((ROOT/'labs/evidence/l169/cost-decision.json').read_text())['decision']=='PROCEED'
    reserve(path,ROOT,phase+'-1',seconds)
    receipt=probe.remote() if phase=='pilot' else remaining.remote()
    print(json.dumps(dict(phase=phase,runs=len(receipt['records']))))

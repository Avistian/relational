"""Approved complete trial experiment; reserve every attempt before dispatch."""
import json,sys,time,traceback
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'labs'))
RATE=.00028372
app=modal.App('l168-cross-database')
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
       .pip_install('numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','pydantic==1.10.26','pyyaml==6.0.2','tabicl==0.1.3')
       .add_local_dir(ROOT/'labs/sources/l166/upstream/model_pretrain','/source/model_pretrain')
       .add_local_file(ROOT/'labs/_run_l168.py','/work/_run_l168.py')
       .add_local_dir('/tmp/l168-input','/input'))
volume=modal.Volume.from_name('l168-cross-database-evidence',create_if_missing=True)

def execute(phase,seeds):
    import sys
    sys.path.insert(0,'/work')
    from _run_l168 import run168
    out=Path('/evidence')/phase;assert not out.exists(),'Immutable phase'
    out.mkdir();start=time.perf_counter()
    try:return run168('/input','/source',out,seeds)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        (out/'cost.json').write_text(json.dumps(dict(worker_body_seconds=time.perf_counter()-start,rate=RATE)));volume.commit()

@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=600,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def probe():return execute('pilot-1',[0])

@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=7200,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def remaining():return execute('full-1',list(range(1,10)))

@app.local_entrypoint()
def main(phase:str='pilot'):
    from _guard_l166 import reserve
    path=ROOT/'labs/evidence/l168/budget.json';b=json.loads(path.read_text())
    seconds=600 if phase=='pilot' else 7200
    assert sum(r['seconds'] for r in b['reservations'])+seconds<=21600,'Aggregate worker-time cap'
    if phase=='pilot':reserve(path,ROOT,'pilot-1',600);print(probe.remote())
    elif phase=='full':
        decision=json.loads((ROOT/'labs/evidence/l168/cost-decision.json').read_text());assert decision['decision']=='PROCEED'
        reserve(path,ROOT,'full-1',7200);print(remaining.remote())
    else:raise ValueError(phase)

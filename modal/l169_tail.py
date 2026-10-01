"""Nonblocking bounded completion; nine missing rows plus six saved consistency sentinels."""
import json,sys,time,traceback
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'labs'))
app=modal.App('l169-context-tail')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install('numpy==1.26.4','pandas==2.2.3','scikit-learn==1.6.1','pydantic==1.10.26','pyyaml==6.0.2','tabicl==0.1.3')
 .add_local_dir(ROOT/'labs/sources/l166/upstream/model_pretrain','/source/model_pretrain')
 .add_local_file(ROOT/'labs/_run_l169_tail.py','/work/_run_l169_tail.py').add_local_dir('/tmp/l169-input','/input'))
volume=modal.Volume.from_name('l169-context-scaling-evidence')
@app.function(image=image,gpu='L4',cpu=(2,2),memory=(16384,16384),timeout=300,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def finish():
 import sys
 sys.path.insert(0,'/work')
 from _run_l169_tail import run169_tail
 out=Path('/evidence/tail-1');assert not out.exists();out.mkdir();start=time.perf_counter()
 jobs={(db,arm,64,0) for db in ['rel-f1','rel-trial'] for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1']}
 jobs|={('rel-trial','TabICLv1.1',1024,s) for s in range(1,10)}
 try:return run169_tail('/input','/source',out,'remaining',jobs)
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  (out/'cost.json').write_text(json.dumps(dict(worker_body_seconds=time.perf_counter()-start,rate=.00028372)));volume.commit()
@app.local_entrypoint()
def main():
 from _guard_l166 import reserve
 path=ROOT/'labs/evidence/l169/budget.json';b=json.loads(path.read_text());assert sum(r['seconds'] for r in b['reservations'])+300<=21600
 reserve(path,ROOT,'tail-1',300);call=finish.spawn()
 (ROOT/'labs/evidence/l169/tail-call.json').write_text(json.dumps(dict(call_id=call.object_id))+'\n');print(call.object_id)

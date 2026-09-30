"""USD10 aggregate, immutable attempts, no automatic retries."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l141-relgnn')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html'))
for p in ['_full_l141.py','_parity_l141.py','_check_l141.py','relkit/relgnn_l141.py']:image=image.add_local_file(ROOT/'labs'/p,'/work/'+p)
image=image.add_local_dir(ROOT/'labs/sources/l141','/work/sources/l141')
volume=modal.Volume.from_name('l141-relgnn-evidence',create_if_missing=True)
RATE=.00022572

image=image.add_local_file(ROOT/'labs/_compat_l141.py','/work/_compat_l141.py')
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=600,retries=0,volumes={'/evidence':volume})
def compat_worker():
 import sys,time,json,traceback,uuid
 sys.path.insert(0,'/work');from _compat_l141 import prepare_checkpoint_compatibility
 from _full_l141 import full_run,sha
 root=Path('/evidence');start=time.perf_counter()
 marker=root/'replay-compatible-started.json';assert not marker.exists();marker.write_text(json.dumps(dict(uuid=str(uuid.uuid4()),trainer_sha256=sha('/work/_full_l141.py'),model_sha256=sha('/work/relkit/relgnn_l141.py'))));volume.commit()
 try:
  info=prepare_checkpoint_compatibility(root/'prepared',root/'compatible')
  r=full_run(root/'replay-compatible',root/'compatible','/work/sources/l141',replay=True)
  r['kind']='CHECKPOINT_COMPATIBILITY_REPLAY';r['compatibility']=info['compatibility']
  (root/'replay-compatible/result.json').write_text(json.dumps(r,indent=2));return r
 except Exception:
  (root/'replay-compatible-failure.txt').write_text(traceback.format_exc());raise
 finally:
  (root/'replay-compatible-cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,worker_body_usd=(time.perf_counter()-start)*RATE)));volume.commit()
@app.local_entrypoint()
def run_compat():
 from l141_repro import reserve
 reserve('replay-compatible',seconds=600);print(compat_worker.remote())

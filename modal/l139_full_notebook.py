"""Execute the portable full notebook gate: fresh preparation and five full fits."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];app=modal.App('l139-full-notebook')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124').pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt').pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html').pip_install('IPython==8.31.0'))
image=image.add_local_file(ROOT/'labs/_full_l139.py','/work/_full_l139.py').add_local_file(ROOT/'labs/relkit/rdl_l117.py','/work/relkit/rdl_l117.py').add_local_file(ROOT/'labs/sources/l139/examples__model.py','/work/sources/l139/examples__model.py')
if (ROOT/'labs/solutions/0139-healthcare-trial.ipynb').exists():image=image.add_local_file(ROOT/'labs/solutions/0139-healthcare-trial.ipynb','/work/lesson.ipynb')
v=modal.Volume.from_name('l139-full-notebook',create_if_missing=True);RATE=.000164+2*.0000131+64*.00000222
@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=1800,retries=0,volumes={'/validation':v})
def prepare():
 import sys,time,json,hashlib,traceback,os
 sys.path.insert(0,'/work');root=Path('/validation');marker=root/'prepare-started.json';assert not marker.exists(),'No automatic restart';marker.write_text(json.dumps(dict(started=time.time())));v.commit()
 start=time.perf_counter()
 try:
  from _full_l139 import materialize
  materialize(root/'l139-prepared')
  report=dict(status='PASS',seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path('/work/_full_l139.py').read_bytes()).hexdigest(),preprocessing='FRESH_FULL_MATERIALIZATION')
  (root/'preparation.json').write_text(json.dumps(report,indent=2));return report
 except Exception:
  (root/'prepare-failure.txt').write_text(traceback.format_exc());raise
 finally:
  (root/'prepare-cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,resource_usd=(time.perf_counter()-start)*RATE)));v.commit()
@app.function(image=image,gpu='T4',cpu=2,memory=65536,timeout=1800,retries=0,volumes={'/validation':v})
def check():
 import json,time,hashlib,os,traceback
 from IPython.display import display
 start=time.perf_counter();root=Path('/validation');marker=root/'started.json';assert not marker.exists(),'No automatic restart'
 marker.write_text(json.dumps(dict(started=time.time())));v.commit();os.chdir(root)
 try:
  notebook=json.loads(Path('/work/lesson.ipynb').read_text());cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'];digest=hashlib.sha256('\n\n'.join(cells).encode()).hexdigest();ns={'display':display,'__file__':'/work/standalone.py'}
  for i,code in enumerate(cells):
   if code.startswith('# Full protocol:'):
    ns['RUN_FULL_REPRODUCTION']=True;ns['PREPARED_ROOT']=root/'l139-prepared'
    preparation=json.loads((root/'preparation.json').read_text());assert preparation['status']=='PASS'
    assert preparation['source_sha256']==hashlib.sha256(Path('/work/_full_l139.py').read_bytes()).hexdigest()
   exec(compile(code,f'cell-{i}','exec'),ns)
   (root/'progress.json').write_text(json.dumps(dict(completed_cell=i,seconds=time.perf_counter()-start)));v.commit()
  # Independent sklearn scores on every newly produced prediction.
  import numpy as np
  from sklearn.metrics import roc_auc_score
  scores={}
  for seed in range(5):
   path=root/'l139-full'/f'seed-{seed}';result=json.loads((path/'result.json').read_text());arrays=np.load(path/'predictions.npz');assert result['epochs']==20 and len(result['history'])==20
   for split in ['val','test']:
    score=roc_auc_score(arrays[split+'_target'],arrays[split+'_pred']);assert abs(score-result['scores'][split]['roc_auc'])<1e-12;scores[f'{seed}-{split}']=score
  report=dict(status='PASS',code_sha256=digest,code_cells=len(cells),seconds=time.perf_counter()-start,full_training='FIVE_FRESH20EPOCH_FITS',preprocessing='FRESH_FULL_MATERIALIZATION_IN_SEPARATE_PHASE',preparation=preparation,scores=scores,report=json.loads((root/'l139-report.json').read_text()),live_colab='NOT_CHECKED')
  (root/'result.json').write_text(json.dumps(report,indent=2));return report
 except Exception:
  (root/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  (root/'cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,resource_usd=(time.perf_counter()-start)*RATE)));v.commit()
@app.local_entrypoint()
def main(mode:str="prepare"):
 import json,fcntl,hashlib
 with (ROOT/'labs/_budget_l139.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);phase='full_notebook_fresh_'+mode;assert not any(r['phase']==phase for r in b['reservations']);upper=1800*RATE
  assert sum(r['upper_usd'] for r in b['reservations'])+upper+3<=10
  b['reservations'].append(dict(phase=phase,upper_usd=upper,timeout=1800,source_sha256=hashlib.sha256((ROOT/'modal/l139_full_notebook.py').read_bytes()).hexdigest()));f.seek(0);json.dump(b,f,indent=2);f.truncate()
 assert mode in ['prepare','check']
 print(prepare.remote() if mode=='prepare' else check.remote())

"""Execute every standalone notebook cell, including full training, in pinned runtime."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];RATE=.00022572
app=modal.App('l149-portable-full-validation')
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('torch==2.5.1',index_url='https://download.pytorch.org/whl/cu124')
 .pip_install_from_requirements(ROOT/'labs/requirements-l117-runtime.txt')
 .pip_install('pyg_lib==0.4.0+pt25cu124',find_links='https://data.pyg.org/whl/torch-2.5.1+cu124.html')
 .pip_install('ipython==8.31.0')
 .add_local_file(ROOT/'labs/solutions/0149-weakest-relbench-tasks.ipynb','/input/solution.ipynb'))
volume=modal.Volume.from_name('l149-notebook-validation-final',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=16384,timeout=900,retries=0,volumes={'/validation':volume})
def worker(expected):
 import os,time,json,hashlib,traceback
 start=time.perf_counter();out=Path('/validation');work=out/'work';assert not work.exists();work.mkdir();os.chdir(work)
 notebook=json.loads(Path('/input/solution.ipynb').read_text());code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
 digest=hashlib.sha256('\n\n'.join(code).encode()).hexdigest();assert digest==expected
 ns={'__name__':'__main__'};count=0
 try:
  for i,source in enumerate(code):
   exec(compile(source.replace('RUN_FULL_GNN_REPRODUCTION = False','RUN_FULL_GNN_REPRODUCTION = True'),f'cell-{i}','exec'),ns);count+=1
  packet=json.loads((work/'l149-full/packet.json').read_text());assert packet['final_fits']==5
  r=dict(status='PASS',code_cells=count,code_sha256=digest,full_gate=True,packet=packet,seconds=time.perf_counter()-start,execution='All standalone notebook code in isolated pinned T4 namespace; not live Colab frontend')
  r['resource_usd']=r['seconds']*RATE;(out/'report.json').write_text(json.dumps(r,indent=2));return r
 except Exception:
  (out/'failure.json').write_text(json.dumps(dict(status='FAIL',completed_cells=count,traceback=traceback.format_exc()),indent=2));raise
 finally:volume.commit()
@app.local_entrypoint()
def main():
 import sys,json,hashlib,fcntl,datetime
 sys.path.insert(0,str(ROOT/'labs'));from relkit.tuning_l135 import reserve_budget
 nb=json.loads((ROOT/'labs/solutions/0149-weakest-relbench-tasks.ipynb').read_text());code='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');digest=hashlib.sha256(code.encode()).hexdigest()
 with (ROOT/'labs/_budget_l149.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  for p,sha in b['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha
  assert not any(x['phase']=='notebook-final-validation' for x in b['reservations'])
  total=reserve_budget([x['upper_usd'] for x in b['reservations']],1,900,RATE,3,10)
  b['reservations'].append(dict(phase='notebook-final-validation',workers=1,upper_usd=900*RATE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=digest));b['total_reserved_usd']=total
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 print(worker.remote(digest))

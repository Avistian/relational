"""Approved full California release experiment, capped reservations, no source edits."""
import hashlib,io,json,os,subprocess,sys,time,zipfile,fcntl
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'labs/evidence/b02';SRC=ROOT/'labs/sources/b02/upstream'
app=modal.App('b02-tabpack-california')
image=(modal.Image.debian_slim(python_version='3.12').apt_install('git').pip_install('uv==0.8.22')
 .add_local_dir(SRC,'/source',copy=True,ignore=['.git','data','__pycache__','.cache'])
 .run_commands('cd /source && UV_PYTHON_PREFERENCE=only-system uv sync --frozen --no-dev --python /usr/local/bin/python')
 .add_local_dir(SRC/'data/california','/source/data/california')
 .add_local_file(ROOT/'labs/_worker_b02.py','/source/worker.py'))
volume=modal.Volume.from_name('b02-tabpack-california',create_if_missing=True)
def execute(phase,limit):
 out=Path('/evidence');receipt=out/(phase+'-receipt.json')
 if receipt.exists():raise RuntimeError('Attempt already exists')
 start=time.monotonic();code=124
 try:
  with (out/(phase+'.log')).open('w') as log:
   p=subprocess.Popen(['/source/.venv/bin/python','/source/worker.py',phase],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(p.pid,signal.SIGKILL);p.wait()
 finally:
  r=dict(phase=phase,exit_code=code,worker_seconds=time.monotonic()-start,timeout_seconds=limit)
  receipt.write_text(json.dumps(r,indent=2));volume.commit()
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file() and p.suffix in ('.json','.npz','.log','.toml'):
    z.write(p,str(p.relative_to(out)))
 return {'receipt':r,'archive':buf.getvalue()}
@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(24576,24576),timeout=660,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def pilot():return execute('pilot3',600)
@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(24576,24576),timeout=1860,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def full():return execute('full',1800)
@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(24576,24576),timeout=1860,retries=0,scaledown_window=2,volumes={'/evidence':volume})
def audited():return execute('audited',1800)
@app.local_entrypoint()
def main(phase:str='pilot'):
 sys.path.insert(0,str(ROOT/'labs'))
 from relkit.embeddings_b02 import reserve_budget
 if phase not in ('pilot','full','audited'):raise ValueError(phase)
 lock=json.loads((E/'source-lock.json').read_text())
 for name,h in lock['source_files'].items():
  assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==h,name
 for name,h in lock['data_files'].items():
  assert hashlib.sha256((SRC/'data'/name).read_bytes()).hexdigest()==h,name
 if phase in ('full','audited'):assert json.loads((E/'admission.json').read_text())['decision']=='PROCEED'
 with (E/'budget.json').open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f)
  b=reserve_budget(b,'pilot3' if phase=='pilot' else phase,660 if phase=='pilot' else 1860,b['rate_usd_second'])
  f.seek(0);json.dump(b,f,indent=2);f.truncate()
 result={'pilot':pilot,'full':full,'audited':audited}[phase].remote()
 (E/(phase+'-raw.zip')).write_bytes(result['archive'])
 with zipfile.ZipFile(io.BytesIO(result['archive'])) as z:z.extractall(E/'fresh')
 (E/(phase+'-receipt.json')).write_text(json.dumps(result['receipt'],indent=2)+'\n')
 print(json.dumps(result['receipt']))

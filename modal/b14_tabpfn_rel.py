"""Budget-reserved original selected release experiment. No source patches."""
import fcntl,hashlib,io,json,os,subprocess,time,zipfile
from pathlib import Path
import modal
R=Path(__file__).resolve().parents[1];E=R/'labs/evidence/b14';SRC=R/'labs/sources/b14/relarena'
app=modal.App('b14-tabpfn-rel-f1')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.13.0','numpy==2.5.3','pandas==2.3.3','scikit-learn==1.6.1','relbench==2.1.2','fastdfs==1.1','tabpfn==8.0.8','setuptools==80.9.0','configspace','jsonschema','pyyaml')
 .env({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TABPFN_DISABLE_TELEMETRY':'1'})
 .add_local_dir(SRC,'/source',ignore=['__pycache__'])
 .add_local_dir('/tmp/b14-dfs-cache','/dfs-cache')
 .add_local_dir(Path.home()/'.cache/relbench/rel-f1','/root/.cache/relbench/rel-f1')
 .add_local_dir('/tmp/b14-weights','/root/.cache/tabpfn')
 .add_local_file(R/'labs/_worker_b14.py','/worker.py')
 .add_local_file(E/'cloud-input-lock.json','/input-lock.json'))
@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(24576,24576),timeout=360,retries=0,scaledown_window=2)
def pilot():return execute('pilot',300)
@app.function(image=image,gpu='A100-80GB',cpu=(4,4),memory=(24576,24576),timeout=1260,retries=0,scaledown_window=2)
def full():return execute('full',1200)
def execute(phase,limit):
 import signal
 out=Path('/out');out.mkdir(exist_ok=True);start=time.monotonic()
 with (out/'worker.log').open('w') as log:
  p=subprocess.Popen(['python','/worker.py',phase],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=124
 (out/'receipt.json').write_text(json.dumps(dict(phase=phase,exit_code=code,seconds=time.monotonic()-start,limit_seconds=limit)))
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  for f in out.iterdir():
   if f.is_file():z.write(f,f.name)
 return b.getvalue()
@app.local_entrypoint()
def main(phase:str='pilot'):
 assert phase in ('pilot','full')
 assert json.loads((E/'paper-preflight.json').read_text())['status']=='PASS'
 assert json.loads((E/'warm-receipt.json').read_text())['status']=='PASS'
 assert json.loads((E/'checkpoint.json').read_text())['status']=='AUTHENTICATED'
 if phase=='full':assert json.loads((E/'admission.json').read_text())['decision']=='PROCEED'
 ledger=E/'cloud-budget.json'
 with ledger.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);seconds=360 if phase=='pilot' else 1260;cost=seconds*b['rate_usd_second']
  assert not any(x['phase']==phase for x in b['reservations']),'No implicit retries'
  assert b['overhead_reserve_usd']+sum(x['usd'] for x in b['reservations'])+cost<=b['stop_usd']
  b['reservations'].append(dict(phase=phase,seconds=seconds,usd=cost));f.seek(0);json.dump(b,f,indent=2);f.truncate()
 raw={'pilot':pilot,'full':full}[phase].remote();(E/(phase+'-raw.zip')).write_bytes(raw)
 target=E/phase;target.mkdir(exist_ok=True)
 with zipfile.ZipFile(io.BytesIO(raw)) as z:z.extractall(target)
 print((target/'receipt.json').read_text())

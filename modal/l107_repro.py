"""Bounded Lesson 107 runs. No automatic retries; aggregate budget recorded in labs/_budget_l107.json."""
from pathlib import Path
import modal
R=Path(__file__).resolve().parents[1]
app=modal.App('l107-snapshot-reproduction')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.3.2','scikit-learn==1.7.1','PyYAML==6.0.2')
 .add_local_dir(R/'labs/relkit','/work/labs/relkit')
 .add_local_dir(R/'labs/sources/l107','/work/labs/sources/l107')
 .add_local_file(R/'labs/_run_sbm_l107.py','/work/labs/_run_sbm_l107.py')
 .add_local_file(R/'labs/_run_wiki_l107.py','/work/labs/_run_wiki_l107.py')
 .add_local_file(R/'labs/_source_check_l107.py','/work/labs/_source_check_l107.py')
 .add_local_file(R/'labs/_audit_l107.py','/work/labs/_audit_l107.py')
 .add_local_file(R/'labs/_analyze_l107.py','/work/labs/_analyze_l107.py')
 .add_local_file(R/'labs/_replay_source_l107.py','/work/labs/_replay_source_l107.py')
 .add_local_file(R/'labs/data/l107/sbm.csv','/work/labs/data/l107/sbm.csv')
 .add_local_file(R/'labs/data/l102/wikipedia.csv','/work/labs/data/l102/wikipedia.csv')
 .add_local_file(R/'labs/data/l102/processed.npz','/work/labs/data/l102/processed.npz'))
volume=modal.Volume.from_name('l107-snapshot-evidence',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=12288,timeout=7500,retries=0,volumes={'/evidence':volume})
def run(lane,arm,seed=0,pilot=False,run_id=""):
 import subprocess,sys,time,json,os,hashlib
 start=time.perf_counter();env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
 digest=hashlib.sha256(b''.join(Path('/work/labs/relkit/'+x).read_bytes() for x in ['snapshot_l107.py','sbm_l107.py','wiki_snapshot_l107.py','tgn_l102.py'])).hexdigest()
 root=Path('/evidence')/digest
 if run_id:
  assert len(run_id)<=64 and all(c.isalnum() or c in '-_' for c in run_id),'Use letters, digits, hyphen or underscore in run ID'
  root=root/'reruns'/run_id
 root=root/('pilot' if pilot else 'full');root.mkdir(parents=True,exist_ok=True)
 seconds=600 if pilot else (7200 if lane=='sbm' else 1800)
 cmd=[sys.executable,f'/work/labs/_run_{lane}_l107.py','--device','cuda','--output',str(root/lane)]
 if lane=='sbm':cmd+=['--variant',arm,'--max-seconds',str(seconds-180)]
 else:cmd+=['--arm',arm,'--seed',str(seed)]
 if pilot:cmd+=['--pilot']
 try:
  subprocess.run([sys.executable,'/work/labs/_source_check_l107.py'],env=env,check=True,timeout=60)
  subprocess.run([sys.executable,'/work/labs/_audit_l107.py'],env=env,check=True,timeout=60)
  subprocess.run(cmd,env=env,check=True,timeout=seconds)
 finally:
  (root/f'call-{lane}-{arm}-{seed}-{time.time_ns()}.json').write_text(json.dumps({'lane':lane,'arm':arm,'seed':seed,'pilot':pilot,'seconds':time.perf_counter()-start,'cutoff':seconds,'digest':digest},indent=2));volume.commit()
 return {'path':str(root),'seconds':time.perf_counter()-start}
@app.local_entrypoint()
def main(lane:str='sbm',arm:str='H',seed:int=0,pilot:bool=False,run_id:str=''):
 print(run.remote(lane,arm,seed,pilot,run_id))
@app.local_entrypoint()
def full_sbm(run_id:str=''):
 for r in run.starmap([('sbm',v,0,False,run_id) for v in ['H','O']]):print(r)
@app.local_entrypoint()
def full_wiki(run_id:str=''):
 for r in run.starmap([('wiki',a,s,False,run_id) for a in ['tgn','3600','86400'] for s in range(3)]):print(r)

@app.function(image=image,gpu='T4',cpu=2,memory=12288,timeout=600,retries=0,volumes={'/evidence':volume})
def source_replay(variant,run_id=""):
 import subprocess,sys,time,json
 start=time.perf_counter();root=Path('/evidence/f31332e889f5d2578543a11ef666ea56311f802e15ea0778cbe4c26db344f447')
 if run_id:
  assert len(run_id)<=64 and all(c.isalnum() or c in '-_' for c in run_id)
  root=root/'reruns'/run_id
 root=root/'full/sbm'/variant
 try:
  subprocess.run([sys.executable,'/work/labs/_replay_source_l107.py','--root',str(root),'--variant',variant,'--device','cuda'],check=True,timeout=540)
 finally:
  (root/f'replay-call-{time.time_ns()}.json').write_text(json.dumps({'seconds':time.perf_counter()-start},indent=2));volume.commit()
 return json.loads((root/'source_replay.json').read_text())
@app.local_entrypoint()
def replay(variant:str='both',run_id:str=''):
 if variant not in ['H','O','both']:raise ValueError('Expected H, O or both')
 for result in source_replay.starmap([(v,run_id) for v in (['H','O'] if variant=='both' else [variant])]):print(result)

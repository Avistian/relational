"""Same visible TGAT on T4; ten independent full Wikipedia runs.
.venv/bin/modal run --detach modal/l103_paper_repro.py::main --preset paper
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l103-tgat-wikipedia')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.3.2','scikit-learn==1.7.1')
 .add_local_file(ROOT/'labs/relkit/tgat_l103.py','/work/relkit/tgat_l103.py')
 .add_local_file(ROOT/'labs/_fetch_l103.py','/work/_fetch_l103.py')
 .add_local_file(ROOT/'labs/_replay_source_l103.py','/work/_replay_source_l103.py')
 .add_local_file(ROOT/'labs/_sources_l103.json','/work/_sources_l103.json')
 .add_local_file(ROOT/'labs/_source_check_l103.py','/work/_source_check_l103.py'))
volume=modal.Volume.from_name('l103-tgat-evidence',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=86400,volumes={'/evidence':volume},max_containers=10)
def experiment(preset,seed):
 import sys,json,hashlib,subprocess,torch,numpy,pandas,sklearn,platform
 sys.path.insert(0,'/work')
 from relkit.tgat_l103 import load_wikipedia,run_training
 torch.set_num_threads(1)
 # Independent raw cache per container avoids partial concurrent downloads.
 n,e,d,audit=load_wikipedia('/tmp/data')
 if preset!='paper':
  tr,ev=(400,120) if preset=='smoke' else (6000,600)
  d={k:({f:x[:tr if k=='train' else ev] for f,x in rows.items()} if k!='full' else rows) for k,rows in d.items()}
 sha=hashlib.sha256(Path('/work/relkit/tgat_l103.py').read_bytes()).hexdigest()
 out=Path('/evidence')/sha/preset/f'seed-{seed}';out.mkdir(parents=True,exist_ok=True)
 identity={'implementation_sha256':sha,'data':audit,'preset':preset,'seed':seed,'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,'device':torch.cuda.get_device_name(),'platform':platform.platform()}
 f=out/'identity.json'
 if f.exists():assert json.loads(f.read_text())==identity,'Identity mismatch'
 else:f.write_text(json.dumps(identity,indent=2))
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 subprocess.run([sys.executable,'/work/_source_check_l103.py','cuda'],check=True)
 (out/'source_parity.json').write_bytes(Path('/work/_source_check_l103_results.json').read_bytes())
 try:
  result=run_training(n,e,d,seed=seed,epochs={'paper':50,'closer':3,'smoke':1}[preset],output=out,device='cuda')
  return {'path':str(out),'seed':seed,'test':result['test'],'new_test':result['new_test']}
 finally:volume.commit()
@app.local_entrypoint()
def main(preset:str='smoke',seeds:str=''):
 if preset not in ['paper','closer','smoke']:raise ValueError(preset)
 selected=list(map(int,seeds.split(','))) if seeds else (list(range(10)) if preset=='paper' else [0])
 for row in experiment.starmap([(preset,seed) for seed in selected]):print(row)

@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=3600,volumes={'/evidence':volume})
def replay_saved():
 import sys,hashlib,torch
 sys.path.insert(0,'/work')
 from _replay_source_l103 import replay
 torch.set_num_threads(1)
 sha=hashlib.sha256(Path('/work/relkit/tgat_l103.py').read_bytes()).hexdigest()
 report=replay(Path('/evidence')/sha/'paper/seed-0','/tmp/data','cuda')
 volume.commit();return report
@app.local_entrypoint()
def source_check():
 print(replay_saved.remote())

"""Optional unattended GPU track; same visible implementation, separate environment.
Run: .venv/bin/modal run --detach modal/l102_paper_repro.py::main --preset paper
Results persist in the l102-tgn-evidence volume. A GPU run is not a CPU replay.
"""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('l102-tgn-wikipedia')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.8.0','numpy==2.2.6','pandas==2.3.2','scikit-learn==1.7.1')
       .add_local_file(ROOT/'labs/relkit/tgn_l102.py','/work/tgn_l102.py')
       .add_local_dir(ROOT/'labs/sources/l102','/work/sources/l102')
       .add_local_file(ROOT/'labs/_source_check_l102.py','/work/_source_check_l102.py')
       .add_local_file(ROOT/'labs/_replay_source_l102.py','/work/_replay_source_l102.py')
       .add_local_file(ROOT/'labs/relkit/tgn_l102.py','/work/relkit/tgn_l102.py'))
volume=modal.Volume.from_name('l102-tgn-evidence',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=86400,volumes={'/evidence':volume})
def experiment(preset,seed_group="0"):
 import sys,json,hashlib,torch,subprocess,shutil,platform,numpy,sklearn,pandas
 sys.path.insert(0,'/work')
 from tgn_l102 import load_wikipedia,run_training
 torch.set_num_threads(1)
 n,e,d,a=load_wikipedia('/evidence/data')
 if preset=='smoke':d={k:{f:x[:200] for f,x in row.items()} if k!='full' else row for k,row in d.items()}
 sha=hashlib.sha256(Path('/work/tgn_l102.py').read_bytes()).hexdigest()
 out=Path('/evidence')/sha/preset/('group-'+seed_group.replace(',','-'));out.mkdir(parents=True,exist_ok=True)
 identity={'source':sha,'data':a,'torch':torch.__version__,'device':torch.cuda.get_device_name(),'python':sys.version,'numpy':numpy.__version__,'sklearn':sklearn.__version__,'pandas':pandas.__version__,'platform':platform.platform()}
 if (out/'identity.json').exists():
  assert json.loads((out/'identity.json').read_text())==identity, 'Existing run has a different environment/data identity'
 else:
  (out/'identity.json').write_text(json.dumps(identity,indent=2))
 subprocess.run([sys.executable,'/work/_source_check_l102.py','cuda'],check=True)
 shutil.copy('/work/_source_check_l102_results.json',out/'source_parity.json')
 records=[]
 try:
  for seed in map(int,seed_group.split(',')):
   done=out/f'seed-{seed}.json'
   records.append(json.loads(done.read_text()) if done.exists() else run_training(n,e,d,seed=seed,epochs=50 if preset=='paper' else 1,output=out,device='cuda'))
   volume.commit()
 finally:volume.commit()
 return {'path':str(out),'scores':[{k:r[k] for k in ['seed','test','new_test']} for r in records]}
@app.local_entrypoint()
def main(preset:str='smoke',seeds:str=''):
 if preset not in ['smoke','paper']:raise ValueError(preset)
 groups=seeds.split(',') if seeds else (['0,1,2','3,4,5','6,7','8,9'] if preset=='paper' else ['0'])
 for result in experiment.starmap([(preset,g) for g in groups]):print(result)

@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=1200,volumes={'/evidence':volume})
def verify_saved_seed():
 import sys,hashlib
 sys.path.insert(0,'/work')
 from _replay_source_l102 import replay
 sha=hashlib.sha256(Path('/work/tgn_l102.py').read_bytes()).hexdigest()
 result=replay(Path('/evidence')/sha/'paper/group-0-1-2','/evidence/data','cuda',0)
 volume.commit()
 return result

@app.local_entrypoint()
def source_check():
 print(verify_saved_seed.remote())

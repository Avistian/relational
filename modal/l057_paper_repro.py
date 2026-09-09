"""L057 three-family OOF stack scale-up; INCOMPARABLE to TabArena Figure 6."""
from pathlib import Path
import modal
app=modal.App('relational-l057')
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('numpy==2.2.6','pandas==2.2.3','pyarrow==19.0.1','scipy==1.15.3','scikit-learn==1.6.1',
              'torch==2.7.1','xgboost==3.0.2','tabicl==0.1.4','einops==0.8.2','huggingface-hub==1.30.0')
 .add_local_dir(str(Path(__file__).resolve().parents[1]/'labs'),remote_path='/course/labs',
                ignore=['data/cache/**','solutions/**','html/**','figures/**','*.ipynb','__pycache__/**']))
volume=modal.Volume.from_name('relational-l057',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=10800,volumes={'/results':volume})
def run(preset='closer'):
 import sys,json,os,hashlib
 sys.path.insert(0,'/course/labs');os.chdir('/course/labs')
 from _fetch_l057 import fetch
 from relkit.cross_experiment import run_suite
 if preset not in ('smoke','lab','closer'):raise ValueError('No faithful paper preset is implemented')
 result=run_suite(preset,include_tfm=True,device='cuda',checkpoint=fetch('/results/checkpoint'),output_dir='/results/'+preset)
 result['source_hashes']={p:hashlib.sha256(Path('/course/labs',p).read_bytes()).hexdigest() for p in ['relkit/cross_ensemble.py','relkit/cross_experiment.py','relkit/tabm.py']}
 path='/results/'+preset+'-results.json';Path(path).write_text(json.dumps(result,indent=2));volume.commit()
 return {'artifact':path,'summary':result['summary'],'verdict':result['verdict']}
@app.local_entrypoint()
def main(preset:str='closer'):print(run.remote(preset))

"""Explicit GPU scale-up for L061 v2; no full paper reproduction is implied."""
from pathlib import Path
import modal
app=modal.App('relational-l061-gp-pfn-v2')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('numpy==2.2.6','torch==2.7.1','scipy==1.15.3')
       .add_local_dir(str(Path(__file__).resolve().parents[1]/'labs/relkit'),remote_path='/course/relkit',ignore=['__pycache__/**']))
volume=modal.Volume.from_name('relational-l061-v2-results',create_if_missing=True)
@app.function(image=image,gpu='T4',cpu=4,memory=16384,timeout=14400,volumes={'/results':volume})
def execute(preset='closer'):
    import sys,time
    sys.path.insert(0,'/course')
    from relkit.pfn_l061_v2 import run_experiment
    path=f'/results/{preset}-{time.time_ns()}.json'
    result=run_experiment(preset,path,device='cuda');volume.commit()
    return {'artifact':path,'status':result['status'],'paper_reproduction':result['paper_reproduction']}
@app.local_entrypoint()
def main(preset:str='closer'):
    print(execute.remote(preset))

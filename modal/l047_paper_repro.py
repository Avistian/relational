"""modal run --detach modal/l047_paper_repro.py --preset closer"""
import os
import sys
import modal
sys.path.insert(0,os.path.dirname(__file__))
from common import app, image_gpu, volumes, artifacts, ARTIFACTS_PATH
image=image_gpu.add_local_dir('labs',remote_path='/root/labs',copy=True)

@app.function(image=image,gpu='T4',volumes=volumes,timeout=24*3600,memory=32768)
def run(preset:str='closer'):
    import os,sys
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    sys.path.insert(0,'/root/labs')
    import _paper_repro_l047
    dest=f'{ARTIFACTS_PATH}/l047/{preset}'
    try:
        _paper_repro_l047.main(['--preset',preset,'--output-dir',dest])
    finally:
        artifacts.commit()
    return {'preset':preset,'artifacts':dest}

@app.local_entrypoint()
def main(preset:str='closer'):
    print(run.remote(preset))

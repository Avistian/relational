"""Run the visible L049 implementation on a T4; no job runs on import.

modal run --detach modal/l049_paper_repro.py --preset closer
modal volume get relational-artifacts l049 ./l049-artifacts
"""
import os
import sys
import modal
sys.path.insert(0,os.path.dirname(__file__))
from common import app,image_gpu,volumes,artifacts,ARTIFACTS_PATH
image=image_gpu.add_local_dir('labs',remote_path='/root/labs',copy=True)


@app.function(image=image,gpu='T4',volumes=volumes,timeout=4*3600,memory=16384)
def run(preset:str='closer'):
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    sys.path.insert(0,'/root/labs');os.chdir('/root/labs')
    from _paper_repro_l049 import run as reproduce
    result=reproduce(preset,out=f'{ARTIFACTS_PATH}/l049/{preset}',device='cuda')
    artifacts.commit()
    return {'preset':preset,'verdict':result['verdict'],'artifact_path':f'l049/{preset}'}


@app.local_entrypoint()
def main(preset:str='closer'):
    print(run.remote(preset))

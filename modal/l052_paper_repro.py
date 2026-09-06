"""modal run --detach modal/l052_paper_repro.py --preset closer"""
import os,sys,modal
sys.path.insert(0,os.path.dirname(__file__))
from common import app,image_gpu,volumes,artifacts,ARTIFACTS_PATH
image=image_gpu.add_local_dir('labs',remote_path='/root/labs',copy=True)
@app.function(image=image,gpu='T4',volumes=volumes,timeout=12*3600,memory=16384)
def run(preset:str='closer'):
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    sys.path.insert(0,'/root/labs');os.chdir('/root/labs')
    from _fetch_l052 import fetch
    from _paper_repro_l052 import reproduce
    fetch()
    result=reproduce(preset,out=f'{ARTIFACTS_PATH}/l052/{preset}',device='cuda')
    artifacts.commit();return result['ledger']
@app.local_entrypoint()
def main(preset:str='closer'):print(run.remote(preset))

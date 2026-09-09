"""Explicit CPU scale-up operator for L058–L070; no faithful paper preset claimed."""
from pathlib import Path
import modal
app=modal.App('relational-foundation-sequence')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('numpy==2.2.6','pandas==2.2.3','scipy==1.15.3','scikit-learn==1.6.1',
                    'torch==2.7.1','xgboost==3.0.2','catboost==1.2.8','pyarrow==19.0.1','einops==0.8.1',
                    'requests','huggingface-hub==0.34.4')
       .add_local_dir(str(Path(__file__).resolve().parents[1]/'labs'),remote_path='/course/labs',
                      ignore=['data/cache/**','solutions/**','html/**','figures/**','*.ipynb','__pycache__/**']))
volume=modal.Volume.from_name('relational-foundation-results',create_if_missing=True)

@app.function(image=image,cpu=4,memory=12288,timeout=14400,volumes={'/results':volume})
def execute(lesson=61,preset='closer',current=False):
    import os,sys,json
    sys.path.insert(0,'/course/labs');os.chdir('/course/labs')
    from _run_foundation import run
    path=run(lesson,preset,f'/results/l{lesson:03}-{preset}'+('-current' if current else '')+'.json',current)
    volume.commit();result=json.loads(Path(path).read_text())
    return dict(artifact=str(path),verdict=result.get('verdict'),scope=result.get('scope',result.get('note')))

@app.local_entrypoint()
def main(lesson:int=61,preset:str='closer',current:bool=False):
    print(execute.remote(lesson,preset,current))

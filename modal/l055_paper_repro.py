"""L055 larger protocol comparison. This does not reproduce TabReD Figure 2."""
from pathlib import Path
import modal

app=modal.App('relational-l055')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('numpy==2.2.6','scipy==1.15.3','scikit-learn==1.6.1','torch==2.7.1','xgboost==3.0.2','threadpoolctl==3.6.0')
       .add_local_dir(str(Path(__file__).resolve().parents[1]/'labs'),remote_path='/course/labs',
                      ignore=['data/cache/**','solutions/**','html/**','figures/**','*.ipynb','__pycache__/**']))
volume=modal.Volume.from_name('relational-l055',create_if_missing=True)

@app.function(image=image,cpu=2,memory=8192,timeout=21600,volumes={'/results':volume})
def run(preset='closer'):
    import sys,json,hashlib
    sys.path.insert(0,'/course/labs')
    from _fetch_l055 import fetch
    from relkit.temporal_experiment import run_suite,summarize
    if preset not in ('smoke','closer'):
        raise ValueError('Only smoke and closer are implemented; paper protocol reconstruction is NOT_RUN')
    config=dict(names=('sberbank-housing',),seeds=(0,),train_cap=300,eval_cap=150,epochs=2,trees=10) if preset=='smoke' else dict(split_ids=(0,1,2),train_cap=6000,eval_cap=2000,epochs=64,trees=300)
    result=run_suite(root=fetch('/results/data'),**config);result['summary']=summarize(result)
    result['source_hashes']={p:hashlib.sha256(Path('/course/labs',p).read_bytes()).hexdigest() for p in ['relkit/temporal.py','relkit/temporal_experiment.py','relkit/tabm.py']}
    output=Path('/results')/(preset+'-results.json');output.write_text(json.dumps(result,indent=1));volume.commit()
    return {'artifact':str(output),'verdict':result['verdict'],'summary':result['summary']}

@app.local_entrypoint()
def main(preset:str='closer'):
    print(run.remote(preset))

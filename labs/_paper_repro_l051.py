"""Required larger L051 intervention run; resume complete datasets only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,platform,sys
from pathlib import Path
from threadpoolctl import threadpool_limits
from relkit.bias_interventions import run_experiment,load_task,DATASETS,PRESETS,rank_summary

def reproduce(preset='closer',out='labs/data/cache/l051-closer',device='cpu',runner=None,identity=None):
    runner=run_experiment if runner is None else runner
    root=Path(__file__).parent;cache=root/'data/cache/l051'
    if identity is None:
        identity={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['relkit/bias_interventions.py','relkit/checkpoint.py','_paper_repro_l051.py']}
    dataset_hashes={n:load_task(n,cache)[2]['sha256'] for n in DATASETS}
    import importlib.metadata
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','xgboost']}
    contract=dict(preset=preset,config=PRESETS[preset],identity=identity,data=dataset_hashes,device=device,versions=versions,python=sys.version,platform=platform.machine())
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Changed implementation/data/config/environment: use a new output directory')
    path.write_text(json.dumps(contract,indent=2));combined=None
    with threadpool_limits(limits=1):
        for name in DATASETS:
            file=out/f'{name}.json'
            if file.exists():
                current=json.loads(file.read_text());print('RESUME',name,flush=True)
            else:
                current=runner(preset,names=[name],cache=cache,device=device)
                temp=file.with_suffix('.tmp');temp.write_text(json.dumps(current,indent=2));temp.replace(file)
            if combined is None:combined={**current,'datasets':{},'elapsed_s':0.}
            combined['datasets'].update(current['datasets']);combined['elapsed_s']+=current['elapsed_s']
    combined['statistics']={c:rank_summary(combined['datasets'],c) for c in ['original','rotated','noise','top5','smoothed']}
    combined['contract']=contract
    combined['ledger']={'verified_here':'Source transformation parity and declared local measurements',
      'paper_claim':'CITED, NOT_REPRODUCED: published aggregated sensitivity and tuning curves',
      'scale_up':f'{preset}: INCOMPARABLE; local split/recipe, 2023 release, 3-task subset, no published search reorders'}
    (out/'summary.json').write_text(json.dumps(combined,indent=2));print(combined['ledger']);return combined

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=['smoke','closer','paper'],default='closer');p.add_argument('--out',default='labs/data/cache/l051-closer');p.add_argument('--device',default='cpu');a=p.parse_args()
    reproduce(a.preset,a.out,a.device)

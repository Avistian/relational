"""Same-code scale-up; complete-seed resume with live-code/data/environment identity."""
import argparse,hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
from relkit.tabr import TabRS
from relkit.tabr_experiment import run_suite,seed_interval
from _live_identity_l051 import code_fingerprint
from relkit.paper_repro import PaperTarget,LabFinding,ScaleUpRun,format_ledger

PRESETS={
 'smoke':dict(train_cap=160,eval_cap=64,d=8,m=4,epochs=2,patience=2),
 'closer':dict(train_cap=6000,eval_cap=None,d=64,m=96,epochs=60,patience=12),
 'paper':dict(train_cap=None,eval_cap=None,d=303,m=96,epochs=1000,patience=16,
              lr=.000280842117039655,dropout=.5508268685197037,context_dropout=.23258422826138023)}

def live_identity(model_class,runner):
    objects={model_class.__init__,model_class.forward,runner}
    # Follow only locally defined functions, including the student's current helpers.
    pending=list(objects)
    while pending:
        f=pending.pop()
        for name in f.__code__.co_names:
            value=f.__globals__.get(name)
            if callable(value) and hasattr(value,'__code__') and value.__module__ in (f.__module__,'__main__') and value not in objects:
                objects.add(value);pending.append(value)
    return {f.__qualname__:code_fingerprint(f) for f in sorted(objects,key=lambda v:v.__qualname__)}

def reproduce(preset='closer',out='data/cache/l052-closer',device='cpu',model_class=TabRS,runner=run_suite):
    here=Path(__file__).resolve().parent;root=here/'data/cache/l052'
    files=sorted((root/'california').glob('*.npy'))
    if len(files)!=6:raise RuntimeError('Fetch the author data using _fetch_l052.py first')
    config=PRESETS[preset];seeds=[0] if preset=='smoke' else list(range(15)) if preset=='paper' else [0,1,2]
    contract=dict(preset=preset,config=config,seeds=seeds,identity=live_identity(model_class,runner),
        operator=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},device=device,
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','scikit-learn','xgboost']},python=sys.version,platform=platform.machine())
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cpath=out/'contract.json'
    if cpath.exists() and json.loads(cpath.read_text())!=contract:raise ValueError('Changed code/data/config/environment: choose a new output directory')
    cpath.write_text(json.dumps(contract,indent=2));runs=[];elapsed=0.
    for seed in seeds:
        path=out/f'seed-{seed}.json'
        if path.exists():current=json.loads(path.read_text());print('RESUME seed',seed,flush=True)
        else:
            current=runner(model_class,names=('california',),seeds=(seed,),arms=('TabR-S',),device=device,data_root=root,**config)
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(current,indent=2));temp.replace(path)
        runs.extend(current['results']['california']['runs']['TabR-S']);elapsed+=current['elapsed_seconds']
    summary=seed_interval([r['score'] for r in runs])
    deviations=['No new hyperparameter search','No quantile jitter; modern library versions and torch search',
                'Only California; no 43-task benchmark or ensembles','Finite epoch cap; initialization and batch RNG not matched']
    if preset!='paper':deviations+=['Subsampled training set, smaller width and local optimization recipe','Three seeds rather than 15 (one for smoke)']
    target=PaperTarget('TabR','2307.14338v2','Table 3: TabR-S','California','RMSE',.403,higher_is_better=False,abs_tol=.01,
                       notes='15 single-model seeds; not full TabR .400 or an ensemble score')
    measured=ScaleUpRun('TabR-S','California','RMSE',summary['mean'],summary['sd'],len(seeds),platform.machine()+' '+device,elapsed,deviations,False)
    ledger=format_ledger(title='L052 TabR-S',lab=[LabFinding('Mechanism and local comparison','See _verify_l052_results.json','Three author tasks; local budget')],paper=[(target,measured,'INCOMPARABLE')])
    result=dict(summary=summary,runs=runs,elapsed_seconds=elapsed,contract=contract,ledger=ledger,verdict='INCOMPARABLE',deviations=deviations)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(ledger);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=list(PRESETS),default='closer');p.add_argument('--out',default='labs/data/cache/l052-closer');p.add_argument('--device',default='cpu');a=p.parse_args()
    reproduce(a.preset,a.out,a.device)

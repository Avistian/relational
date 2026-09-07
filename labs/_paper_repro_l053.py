"""Live-code TD-S scale-up. Complete-seed resume; explicit non-paper splits."""
import argparse,hashlib,importlib.metadata,inspect,json,platform,sys
from pathlib import Path
from relkit.realmlp import RealMLPS,RobustSmooth
from relkit.realmlp_experiment import run_suite
from _live_identity_l051 import code_fingerprint
from relkit.paper_repro import PaperTarget,LabFinding,ScaleUpRun,format_ledger

PRESETS={
 'smoke':dict(train_cap=160,eval_cap=64,width=16,epochs=2),
 'closer':dict(train_cap=6000,eval_cap=None,width=256,epochs=256),
 'paper':dict(train_cap=None,eval_cap=None,width=256,epochs=256)}

def identity(*roots):
    """Follow live lesson functions and class methods, including student's helpers."""
    pending=list(roots);seen=set();found={}
    while pending:
        obj=pending.pop()
        if id(obj) in seen:continue
        seen.add(id(obj))
        if inspect.isclass(obj):
            pending.extend(v for v in vars(obj).values() if inspect.isfunction(v))
        elif inspect.isfunction(obj):
            found[obj.__qualname__]=code_fingerprint(obj)
            for name in obj.__code__.co_names:
                value=obj.__globals__.get(name)
                module=getattr(value,'__module__','') or ''
                if (inspect.isclass(value) or inspect.isfunction(value)) and (module=='__main__' or module.startswith('relkit.realmlp')):
                    pending.append(value)
    return dict(sorted(found.items()))

def reproduce(preset='closer',out='data/cache/l053-closer',device='cpu',
              model_class=RealMLPS,runner=run_suite,prep_class=RobustSmooth):
    root=Path(__file__).resolve().parent;data_root=root/'data/cache/l052'
    seeds=[0] if preset=='smoke' else list(range(10)) if preset=='paper' else [0,1,2]
    config=PRESETS[preset]
    files=sorted((data_root/'california').glob('*.npy'))
    if len(files)!=6:raise RuntimeError('Run _fetch_l052.fetch() first')
    contract=dict(preset=preset,config=config,seeds=seeds,identity=identity(model_class,runner,prep_class),
        operator=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},device=device,
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','xgboost']},
        python=sys.version,platform=platform.machine())
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Changed live code/data/config/environment; choose a new output directory')
    path.write_text(json.dumps(contract,indent=2));runs=[];elapsed=0.
    for seed in seeds:
        path=out/f'seed-{seed}.json'
        if path.exists():current=json.loads(path.read_text());print('RESUME',seed,flush=True)
        else:
            current=runner(model_class,names=('california',),seeds=(seed,),device=device,data_root=data_root,
                           neural_only=True,prep_class=prep_class,**config)
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(current,indent=2));tmp.replace(path)
        runs.extend(current['results']['california']['runs']['RealMLP-S']);elapsed+=current['elapsed_seconds']
    from relkit.tabr_experiment import seed_interval
    summary=seed_interval([r['error'] for r in runs])
    gaps=['TabR California release split, not RealMLP ten 60/20/20 splits',
          'Numeric TD-S only; not full TD or the meta-test benchmark',
          'Training-only target standardization, matching standalone rather than paper train+validation wording',
          'No benchmark-wide default optimization, bagging, refit, or boosted-tree HPO reproduction']
    if preset!='paper':gaps+=['Capped training rows; fewer than ten repeats']
    if preset=='smoke':gaps+=['Reduced width and compressed epoch schedule']
    target=PaperTarget('Better by Default','2407.04491v3','§5 / Appendix A.2',
        'California','RMSE',None,notes='Benchmark-level claim has no single California RMSE target.')
    run=ScaleUpRun('Numeric TD-S','California','RMSE',summary['mean'],summary['sd'],len(seeds),device,elapsed,gaps,False)
    ledger=format_ledger(title='L053 strong defaults',lab=[LabFinding('Local comparison','See _verify_l053_results.json','Three fixed TabR-release splits; 3 model seeds; six-candidate XGB search')],paper=[(target,run,'INCOMPARABLE')])
    result=dict(summary=summary,runs=runs,elapsed_seconds=elapsed,contract=contract,verdict='INCOMPARABLE',deviations=gaps,ledger=ledger)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(ledger);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=list(PRESETS),default='closer');p.add_argument('--out',default='labs/data/cache/l053-closer');p.add_argument('--device',default='cpu');a=p.parse_args()
    reproduce(a.preset,a.out,a.device)

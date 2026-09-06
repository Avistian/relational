"""Required next step: the same visible comparison on the paper's Higgs-small task.

All presets remain INCOMPARABLE: fresh 60/20/20 split, standardization, two
candidates, AUROC selection, and no reconstruction of the paper's tuning/ensembles.
"""
import argparse
import hashlib
import inspect
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from relkit.checkpoint import run_comparison, paired_summary, CACHE
from relkit.paper_repro import PaperTarget, ScaleUpRun, LabFinding, format_ledger

PRESETS = {
    'smoke':dict(cap=800,d=16,layers=1,epochs=2,trees=20),
    'closer':dict(cap=6000,d=64,layers=3,epochs=50,trees=400),
    # The name denotes resource scale, not a claim of matching the paper protocol.
    'paper':dict(cap=None,d=192,layers=3,epochs=100,trees=2000),
}
TARGET = PaperTarget(paper='Gorishniy et al. 2021',arxiv='2106.11959v5',
    table='Table 2: FT-T, HI (single models)',dataset='Higgs Small',metric='accuracy',
    paper_value=.729,abs_tol=.01,notes='Paper §4.5 tree comparison is a different ensemble table.')
GAPS = ['Fresh 60/20/20 partition; author split IDs not reconstructed',
        'Median/standard scaling rather than paper neural quantile transform',
        'Two fixed candidates, AUROC selection; no paper hyperparameter search',
        'Three seeds rather than full paper repetition protocol',
        'Local width/dropout/epochs/batch size and modern software differ',
        'Table 4 ensemble comparison is not run']


def run(preset='closer',out='data/cache/l050-closer',device='cpu',experiment=run_comparison,implementation_id=None):
    if experiment is not run_comparison and not implementation_id:
        raise ValueError('A visible notebook experiment must supply its implementation identity.')
    cfg=PRESETS[preset]; folder=Path(out); folder.mkdir(parents=True,exist_ok=True)
    seeds=[0] if preset=='smoke' else [0,1,2]
    source=Path(__file__).resolve().parent/'relkit/checkpoint.py'
    data=CACHE/'higgs_small.parquet'
    if not data.exists():
        from relkit.data import load_tier_a
        load_tier_a('higgs_small')
    identity={'preset':preset,'config':cfg,'device':device,'seeds':seeds,'implementation_id':implementation_id,
        'versions':{p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','scikit-learn','xgboost')},
        'source':hashlib.sha256(source.read_bytes()).hexdigest(),
        'operator':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data':hashlib.sha256(data.read_bytes()).hexdigest()}
    manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Cached experiment identity changed; choose a new output directory.')
    manifest.write_text(json.dumps(identity,indent=2))
    records=[]
    for seed in seeds:
        path=folder/f'seed-{seed}.json'
        if path.exists():
            result=json.loads(path.read_text());print('Resume completed seed',seed)
        else:
            result=experiment(datasets=('higgs_small',),seeds=(seed,),device=device,**cfg)
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
        records.append(result)
    scores={m:[r['datasets']['higgs_small']['runs'][m][0]['accuracy'] for r in records]
            for m in records[0]['models']}
    gaps=GAPS+([f'Label-independent cap={cfg["cap"]} of 98050 rows'] if cfg['cap'] else [])
    summary={'preset':preset,'accuracy':{m:paired_summary(v,np.zeros(len(v))) for m,v in scores.items()},
             'verdict':'INCOMPARABLE','protocol_deviations':gaps,'identity':identity,
             'elapsed_s':sum(r['elapsed_s'] for r in records)}
    measured=summary['accuracy']['FT-Transformer']
    scale=ScaleUpRun(method='from-scratch numeric FT-Transformer',dataset='Higgs Small',metric='accuracy',
        value=measured['mean'],std=measured['sd'],n_seeds=len(seeds),hardware=device,
        wall_s=summary['elapsed_s'],protocol_deviations=gaps,protocol_match=False)
    summary['ledger']=format_ledger(title='L050',lab=[LabFinding('Local checkpoint',
        'see _verify_l050_results.json','three substitute datasets; two candidates per arm')],
        paper=[(TARGET,scale,'INCOMPARABLE')])
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary['ledger'])
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,default='smoke')
    p.add_argument('--out');p.add_argument('--device',default='cpu');a=p.parse_args()
    run(a.preset,a.out or f'data/cache/l050-{a.preset}',a.device)

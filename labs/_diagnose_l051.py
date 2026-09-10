"""Additive Fig.3 bandwidth diagnosis; preserves the historical L051 experiment.

One task, fixed local recipes, raw validation/test targets, no HP search.
This is a measured sensitivity curve, not reproduction of the published curve.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits
from relkit.bias_interventions import prepare_task,smooth_targets,fit_arm,paired_effect,MODELS


def smoothing_sweep(name,cache,max_rows=1800,hs=(0.,.25,.5,1.),seeds=(0,1,2),epochs=20,trees=120,device='cpu'):
    """Refit after each target intervention, pairing recipes, seeds and raw test rows."""
    if not hs or hs[0]!=0 or any(h<0 for h in hs):
        raise ValueError('Start with h=0 as the paired top-five baseline')
    states,labels,metadata=prepare_task(name,cache,max_rows)
    xs=states['top5'][0];cov=np.asarray(metadata['covariance'])
    started=time.perf_counter();rows=[]
    for h in hs:
        _,train_y=smooth_targets(xs[0],labels[0],h,cov)
        row=dict(h=float(h),changed_labels=int(np.sum(train_y!=labels[0])),
                 positive_fraction=float(train_y.mean()),runs={},effects={},status='TRAINED')
        if np.unique(train_y).size<2:
            row['status']='NOT_FIT_ONE_CLASS'
            rows.append(row)
            continue
        for model in MODELS:
            runs=[]
            for seed in seeds:
                p,selection=fit_arm(model,xs,train_y,labels[1],seed,epochs,trees,device)
                runs.append(dict(seed=seed,accuracy=float(accuracy_score(labels[2],p>=.5)),
                                 probability=p.tolist(),**selection))
            row['runs'][model]=runs
        baseline=rows[0] if rows else row
        row['effects']={m:paired_effect([r['accuracy'] for r in row['runs'][m]],
                                       [r['accuracy'] for r in baseline['runs'][m]]) for m in MODELS}
        rows.append(row)
    return dict(dataset=name,verdict='INCOMPARABLE',scope='One-task fixed-recipe bandwidth diagnosis',
                config=dict(max_rows=max_rows,hs=list(hs),seeds=list(seeds),epochs=epochs,trees=trees,device=device),
                split_rows=metadata['split_rows'],test_y=labels[2].tolist(),top5=metadata['top5'],
                covariance=metadata['covariance'],data_sha256=metadata['sha256'],rows=rows,
                elapsed_s=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='electricity')
    parser.add_argument('--out',default='labs/_diagnostic_l051_results.json')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    with threadpool_limits(limits=1):
        result=smoothing_sweep(args.dataset,root/'data/cache/l051')
    import importlib.metadata
    result['versions']={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','xgboost']}
    result['source_sha256']={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                            ['_diagnose_l051.py','relkit/bias_interventions.py','relkit/checkpoint.py']}
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(args.out,'seconds',round(result['elapsed_s']))
    for row in result['rows']:
        print(row['h'],row['changed_labels'],{m:round(e['mean'],5) for m,e in row['effects'].items()})

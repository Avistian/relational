"""SAINT supervised Bank/Table 2 attempt. Presets do not certify a protocol match.
Run: python labs/_paper_repro_l047.py --preset smoke --output-dir /tmp/l047_smoke
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from relkit.saint import SAINT, train_saint, predict_saint
from relkit.saint_experiment import prepare, environment
from relkit.paper_repro import LabFinding, PaperTarget, ScaleUpRun, classify_number, format_ledger

TARGET = PaperTarget(paper='SAINT (Somepalli et al., 2021)',arxiv='2106.01342',
    table='Table 2, Bank, supervised SAINT',dataset='Bank (OpenML 1461)',metric='AUROC',
    paper_value=.9330,abs_tol=.01,
    notes='Five-trial mean; not the 14-dataset mean 0.9313. Table 6 reports SE 0.0009, not SD.')
PRESETS = {
    'smoke':dict(cap=600,epochs=1,d=4,heads=2,seeds=[0],batch_size=32,ff_dropout=.1,lr=.001),
    'closer':dict(cap=None,epochs=20,d=16,heads=4,seeds=[0,1,2],batch_size=256,ff_dropout=.8,lr=.0001),
    'paper':dict(cap=None,epochs=100,d=32,heads=8,seeds=[0,1,2,3,4],batch_size=256,ff_dropout=.8,lr=.0001),
}
GAPS = [
    'Original five split-index files unavailable: released 65/15/20 assignment, split seed 5.',
    'Paper says 65/15/25 (105%); released data_openml.py uses 65/15/20.',
    'Released normalization LN(x)+F(LN(x)), GEGLU, numeric 1→100→d; prose describes different placement/embedding.',
    'Training-only category vocabulary with reserved unknown; released encoder fits vocabulary to full table.',
    'Paper uses 8 heads; current train.py clamps combined variant to 4. paper preset chooses paper count.',
    'Attention dropout declared but unused in released forward; feed-forward dropout is applied.',
    'Bank only; no reproduction of 14-task mean, semi-supervised Table 3, or superiority over tuned trees.',
]


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--preset',choices=PRESETS,default='closer')
    parser.add_argument('--output-dir',default=os.environ.get('L047_OUTPUT','l047_artifacts'))
    args=parser.parse_args(argv); cfg=PRESETS[args.preset]
    dest=Path(args.output_dir); dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    fr=prepare('bank_marketing',cap=cfg['cap'],split='released')
    (dest/'protocol.json').write_text(json.dumps(fr['meta'],indent=2))
    from relkit import saint_experiment
    source_dir = Path(saint_experiment.__file__).resolve().parents[1]
    source_hashes = {name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
                     for name in ('relkit/saint.py','relkit/saint_experiment.py','_paper_repro_l047.py')}
    values=[]; start=time.time()
    for seed in cfg['seeds']:
        artifact=dest/f'{args.preset}_seed{seed}.json'
        identity=hashlib.sha256(json.dumps({'cfg':cfg,'data':fr['meta']['data_sha256'],
            'split':fr['meta']['split_hashes'],'seed':seed,'source_hashes':source_hashes,'environment':environment()},sort_keys=True).encode()).hexdigest()
        if artifact.exists():
            record=json.loads(artifact.read_text())
            if record.get('identity') != identity:
                raise RuntimeError(f'{artifact} belongs to a different run; use a fresh output directory')
        else:
            torch.manual_seed(seed)
            model=SAINT(fr['xn'].shape[1],fr['cards'],d=cfg['d'],heads=cfg['heads'],
                        ff_dropout=cfg['ff_dropout'],variant='colrow')
            model,hist=train_saint(model,fr['xn'],fr['xc'],fr['y'],fr['train'],fr['valid'],seed=seed,
                epochs=cfg['epochs'],batch_size=cfg['batch_size'],lr=cfg['lr'],device=dev,
                checkpoint=str(dest/f'{args.preset}_seed{seed}_best.pt'),
                select_metric='accuracy',validate_every=5)
            te=fr['test']; pred=predict_saint(model,fr['xn'][te],fr['xc'][te],cfg['batch_size'],dev)
            np.savez(dest/f'{args.preset}_seed{seed}_predictions.npz',row_ids=te,y=fr['y'][te],p=pred)
            record={'identity':identity,'seed':seed,'auc':float(roc_auc_score(fr['y'][te],pred)),
                    'history':hist,'environment':environment()}
            artifact.write_text(json.dumps(record,indent=2))
        values.append(record['auc']); print(f'Bank seed {seed}: AUROC {values[-1]:.6f}',flush=True)
    gaps=GAPS+[f'Preset {args.preset}: {cfg}. Selection follows released binary code: validation accuracy every fifth epoch.']
    run=ScaleUpRun(method='from-scratch SAINT',dataset=TARGET.dataset,metric='AUROC',
        value=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None,
        n_seeds=len(values),hardware=str(torch.cuda.get_device_name(0)) if dev=='cuda' else 'CPU',
        wall_s=time.time()-start,protocol_match=False,protocol_deviations=gaps)
    ledger=format_ledger(title='L047 supervised SAINT / Bank',
        lab=[LabFinding('Row-context mechanism and local ablation','See _check_l047_results.json and _verify_l047_results.json',
                        'Three substitute datasets; no pretraining')],
        paper=[(TARGET,run,classify_number(TARGET,run))],
        extra_lines=[f'Descriptive delta from Bank 0.9330: {run.value-.9330:+.6f}; ±0.01 tolerance applies only after protocol agreement.',
                     'Completed seeds resume; interrupted seeds restart. Best weights support inference, not optimizer resume.'])
    out={'preset':args.preset,'source_hashes':source_hashes,'config':cfg,'environment':environment(),'scores':values,'ledger':ledger}
    (dest/f'{args.preset}_results.json').write_text(json.dumps(out,indent=2))
    print(ledger); return out


if __name__ == '__main__':
    main()

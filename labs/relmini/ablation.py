"""Cache-only seed0 constructor feature pair; local evidence, not RelBench-comparable."""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

import baseline
import verify

NOTE = 'LOCAL reproducibility evidence only; not RelBench-comparable'


def run(*, cache=baseline.CACHE, output, smoke=False):
    started = time.monotonic()
    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output must be empty to preserve evidence: {output}")
    receipt = {'status': 'NOT_RUN', 'seed': 0, 'note': NOTE, 'usd': 0,
               'mode': 'tiny cached smoke' if smoke else 'full cached pair',
               'actor': os.environ.get('LEAD_ACTOR', 'local caller'),
               'pane': os.environ.get('HERDR_PANE_ID', 'unavailable')}

    def save():
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')

    pins = {name: {'path': str(cache / name), 'sha256': sha} for name, sha in verify._P.items()}
    problems = verify.check_hashes(pins)
    if problems:
        receipt['reason'] = problems
        save()
        print('NOT_RUN:', '; '.join(problems))
        return 1

    train, results, drivers = baseline.load_real('train', cache)
    constructors = pd.read_parquet(cache / 'db' / 'constructor_results.parquet')
    splits = {'train': train}
    for split in ('val', 'test'):
        splits[split] = baseline.load_real(split, cache)[0]
    if smoke:
        splits = {split: frame.iloc[:min(128, len(frame))].copy() for split, frame in splits.items()}
    if any(frame.empty or frame.did_not_finish.nunique() != 2 for frame in splits.values()):
        receipt['reason'] = 'empty split or missing binary classes'
        save()
        print('NOT_RUN:', receipt['reason'])
        return 1

    # Save exact evaluated label rows (ordered); whole input files remain read-only in cache.
    split_hashes = {}
    for split, frame in splits.items():
        path = output / f'{split}-rows.csv'
        frame.to_csv(path, index=False)
        split_hashes[split] = verify.sha256_file(path)

    metrics, predictions, cutoffs = {}, [], {}
    for split, frame in splits.items():
        feats = baseline.build_features(frame, results, drivers, constructors)
        baseline.assert_no_leakage(feats)
        cutoffs[split] = {col: int(feats[col].notna().sum()) for col in
                         ('feature_cutoff', 'constructor_assignment_cutoff', 'constructor_cutoff')}
        feats[['driverId', 'date', 'feature_cutoff', 'constructor_assignment_cutoff',
               'constructor_cutoff'] + baseline.FEATURES + baseline.CONSTRUCTOR_FEATURES].to_csv(
                   output / f'{split}-features.csv', index=False)

    for variant, enabled in (('baseline_without_constructor', False), ('with_constructor', True)):
        metrics[variant] = {}
        for split in ('val', 'test'):
            ev = splits[split]
            a = baseline.fit_predict(splits['train'], ev, results, drivers, seed=0,
                                     constructor_results=constructors, use_constructor=enabled)
            b = baseline.fit_predict(splits['train'], ev, results, drivers, seed=0,
                                     constructor_results=constructors, use_constructor=enabled)
            if not np.array_equal(a, b):
                raise AssertionError('paired predictions not deterministic')
            metrics[variant][split + '_auroc'] = baseline.auroc(ev.did_not_finish.to_numpy(), a)
            metrics[variant][split + '_n'] = len(ev)
            pred = ev[['driverId', 'date', 'did_not_finish']].reset_index(drop=True).copy()
            pred['split'], pred['variant'] = split, variant
            pred['probability'], pred['repeat_probability'] = a, b
            predictions.append(pred)
    pd.concat(predictions, ignore_index=True).to_csv(output / 'predictions.csv', index=False)
    source = Path(__file__).resolve().parent
    receipt.update(status='PASS', metrics=metrics, input_hashes={n: p['sha256'] for n, p in pins.items()},
                   split_hashes=split_hashes, cutoff_rows_checked=cutoffs, double_run_identical=True,
                   model='StandardScaler+LogisticRegression', hyperparameters={'max_iter': 1000, 'random_state': 0},
                   features={'baseline_without_constructor': baseline.FEATURES,
                             'with_constructor': baseline.FEATURES + baseline.CONSTRUCTOR_FEATURES},
                   versions={'python': platform.python_version(), 'numpy': np.__version__,
                             'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                   source_hashes={p.name: verify.sha256_file(p) for p in source.glob('*.py')},
                   source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
                   elapsed_s=time.monotonic() - started)
    receipt['artifact_hashes'] = {p.name: verify.sha256_file(p) for p in sorted(output.glob('*.csv'))}
    save()  # receipt is the final artifact of a successful run
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='first 128 cached label rows per split')
    args = parser.parse_args()
    raise SystemExit(run(output=args.output, smoke=args.smoke))

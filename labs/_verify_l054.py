"""Run the measured L054 evidence and write _verify_l054_results.json.

Main comparison: MLP vs MLP-x32 vs TabM-mini(k=32) vs XGB-tuned on three cached
numeric tasks (caps 1200/600/600, seeds 0/1/2). Plus a real k-sweep on California
(k in 1..32) to show how the ensemble effect depends on the number of submodels.
All numbers are local protocol evidence; verdict is INCOMPARABLE to the paper.
"""
import json
import time
from pathlib import Path
import torch
from relkit.realmlp_experiment import load_task
from relkit.tabm_experiment import run_suite, _standardise, fit_tabm

ROOT = Path(__file__).resolve().parent


def k_sweep(name='california', ks=(1, 2, 4, 8, 16, 32), seed=0,
            width=64, depth=3, epochs=64, lr=2e-3, dropout=.1, data_root='data/cache/l052'):
    torch.set_num_threads(1)
    data = load_task(name, 1200, 600, data_root)
    xr = _standardise(data['raw'])
    xs = {s: torch.as_tensor(v) for s, v in xr.items()}
    ys = {s: torch.as_tensor(data['y'][s], dtype=torch.float32) for s in xr}
    rows = []
    for k in ks:
        r = fit_tabm(data, xs, ys, seed, k=k, width=width, depth=depth,
                     epochs=epochs, lr=lr, dropout=dropout, arch='mini')
        rows.append(dict(k=k, collective=r['error'], individual_mean=r['individual_mean'],
                         individual_best=r['individual_best']))
        print('k-sweep', name, k, round(r['error'], 5), flush=True)
    return dict(dataset=name, seed=seed, metric='RMSE', rows=rows)


if __name__ == '__main__':
    start = time.time()
    result = run_suite(seeds=(0, 1, 2), k=32, width=64, depth=3, epochs=64, lr=2e-3)
    result['k_sweep'] = k_sweep()
    result['total_seconds'] = time.time() - start
    (ROOT / '_verify_l054_results.json').write_text(json.dumps(result, indent=1))
    print('WROTE _verify_l054_results.json in', round(result['total_seconds'], 1), 's')

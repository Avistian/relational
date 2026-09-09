"""Join measured version panels only after verifying identical dataset contracts."""
import argparse, json
from pathlib import Path
from relkit.benchmark_core import paired_summary

ROOT = Path(__file__).resolve().parent


def assemble(historical, current, output):
    a, b = [json.loads(Path(p).read_text()) for p in [historical, current]]
    if a['datasets'] != b['datasets']:
        raise ValueError('The panels must use identical data and row partitions')
    records = a['records'] + b['records']
    summary = paired_summary(records)  # Reject duplicates and missing crossed cells.
    expected = {'XGBoost', 'TabM-mini', 'TabPFN-v2', 'TabICL-v1.1',
                'TabPFN-2.5-synthetic', 'TabPFN-3', 'TabICLv2'}
    if set(summary['arms']) != expected or len(records) != 105:
        raise ValueError('Require all seven arms on five tasks and three seeds')
    a.update(records=records, summary={'random': summary},
             scope='Seven explicit checkpoint/model arms on five fixed binary tasks, three seeds; restricted CPU budgets',
             historical_summary=a['summary'], current_versions=b['versions'],
             current_checkpoints=b['checkpoints'],
             requested_arms={arm: 'MEASURED' for arm in sorted(expected)},
             verdict='INCOMPARABLE',
             paper_reproduction='NOT_ESTABLISHED')
    Path(output).write_text(json.dumps(a, indent=2, allow_nan=False)+'\n')
    return a


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--historical', default=ROOT/'_verify_l070_historical_results.json')
    p.add_argument('--current', default=ROOT/'_verify_l070_current_results.json')
    p.add_argument('--output', default=ROOT/'_verify_l070_results.json')
    args = p.parse_args()
    assemble(args.historical, args.current, args.output)

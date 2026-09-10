"""Independently reconcile the L055 reanalysis with committed author report blobs."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
REVISION = 'b5ef15b3749f30da7a1eb8fba21a5b54d706bf32'


def check(checkout):
    evidence = json.loads((ROOT / '_paper_l055_results.json').read_text())
    assert evidence['revision'] == REVISION
    tree = subprocess.check_output(['git', '-C', str(checkout), 'ls-tree', '-r', '-z',
                                    REVISION, '--', 'paper/exp/temporal-shift-analysis'])
    blobs = {}
    for entry in tree.split(b'\0'):
        if entry:
            metadata, name = entry.split(b'\t', 1)
            blobs[name.decode()] = metadata.split()[2].decode()
    records = evidence['records']
    names = [r['path'] for r in records]
    assert len(names) == len(set(names)) == 2879
    missing = 'paper/exp/temporal-shift-analysis/xgboost_/cooking-time-random-0/evaluation/1/report.json'
    assert evidence['missing'] == [missing] and missing not in blobs
    assert evidence['expected'] == 2880 and evidence['found'] == len(records)
    payload = ''.join(blobs[name] + '\n' for name in names).encode()
    response = subprocess.run(['git', '-C', str(checkout), 'cat-file', '--batch'],
                              input=payload, stdout=subprocess.PIPE, check=True).stdout
    position = 0
    groups = defaultdict(dict)
    for record in records:
        end = response.index(b'\n', position)
        object_id, kind, size = response[position:end].split()
        assert kind == b'blob' and object_id.decode() == blobs[record['path']]
        assert record['git_blob'] == object_id.decode()
        position = end + 1
        raw = response[position:position + int(size)]
        position += int(size) + 1
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
        source = json.loads(raw)
        config = source['config']
        arm = Path(record['path']).parts[3]
        assert record['model'] == {'mlp': 'MLP', 'mlp-plr': 'MLP-PLR', 'tabr': 'TabR-S', 'xgboost_': 'XGBoost'}[arm]
        assert record['protocol'] in ('random', 'temporal')
        assert record['window'] in range(3) and record['seed'] in range(15)
        expected_split = ('random' if record['protocol'] == 'random' else 'sliding-window') + '-' + str(record['window'])
        assert config['seed'] == record['seed']
        assert config['data']['path'] == ':data/' + record['task']
        assert config['data']['split'] == expected_split
        assert record['value'] == source['metrics']['test'][record['metric']]
        assert record['validation'] == source['metrics']['val'][record['metric']]
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert digest == record['config_sha256']
        key = (record['task'], record['model'], record['protocol'], record['window'])
        assert record['seed'] not in groups[key]
        groups[key][record['seed']] = record['value']
    assert position == len(response)
    comparisons = 0
    for section, matched in [('summary', False), ('matched_seed_summary', True)]:
        assert len(evidence[section]) == 64
        assert len({(s['task'], s['model'], s['protocol']) for s in evidence[section]}) == 64
        for summary in evidence[section]:
            means = []
            for window in range(3):
                key = (summary['task'], summary['model'], summary['protocol'], window)
                seeds = set(groups[key])
                if matched:
                    for other, values in groups.items():
                        if other[0] == summary['task'] and other[3] == window:
                            seeds &= set(values)
                values = [groups[key][seed] for seed in sorted(seeds)]
                mean = statistics.mean(values)
                means.append(mean)
                reported = summary['windows'][window]
                assert reported['n'] == len(values)
                assert math.isclose(reported['mean'], mean, abs_tol=1e-14, rel_tol=0)
                assert math.isclose(reported['seed_sd'], statistics.stdev(values), abs_tol=1e-14, rel_tol=0)
                comparisons += 1
            assert math.isclose(summary['mean'], statistics.mean(means), abs_tol=1e-14, rel_tol=0)
            assert math.isclose(summary['window_min'], min(means), abs_tol=1e-14, rel_tol=0)
            assert math.isclose(summary['window_max'], max(means), abs_tol=1e-14, rel_tol=0)
    report = dict(status='PASS', committed_author_reports=len(records),
                  aggregate_window_cells=comparisons, missing_report=missing,
                  source='Pinned Git objects, independent of working-tree report contents',
                  scope='Author-score extraction and arithmetic reconciliation; no retraining or paper-data equivalence claim')
    (ROOT / '_check_paper_reports_l055_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    check(parser.parse_args().checkout)

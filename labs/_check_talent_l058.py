"""Independently reconstruct L058 reported ranks from pinned Markdown cells."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
FILES = {'binary': 'cls_bin.md', 'multiclass': 'cls_multi.md', 'regression': 'regression.md'}


def check(path):
    result = json.loads(path.read_text())
    tables = {}
    for task, name in FILES.items():
        lines = [line for line in (ROOT / 'sources/l058' / name).read_text().splitlines()
                 if line.startswith('|')]
        header = [x.strip() for x in lines[0].strip('|').split('|')][1:]
        header = [{'LogReg': 'LR', 'LinearRegression': 'LR'}.get(x, x) for x in header]
        ids, rows = [], []
        for line in lines[2:]:
            cells = [x.strip() for x in line.strip('|').split('|')]
            ids.append(cells[0])
            rows.append([float(x.strip('*').split('+')[0]) if x.strip('*') not in ('', '-', '—', 'NA', 'N/A')
                         else np.nan for x in cells[1:]])
        assert len(ids) == len(set(ids))
        tables[task] = (ids, header, np.array(rows))

    def ranks(values, task):
        # Average rank = 1 + number strictly better + half the other ties.
        values = -values if task != 'regression' else values
        return np.array([[1 + sum(row < value) + (sum(row == value) - 1) / 2
                          for value in row] for row in values], dtype=float)

    all_ranks = []
    for task, (ids, names, values) in tables.items():
        values = values[:, [names.index(a) for a in result['arms']]]
        values = values[np.isfinite(values).all(axis=1)]
        r = ranks(values, task)
        assert len(r) == result['coverage'][task]['complete']
        np.testing.assert_allclose(r.mean(0), list(result['per_task_ranks'][task].values()), atol=1e-14, rtol=0)
        all_ranks.extend(r)
    all_ranks = np.array(all_ranks)
    np.testing.assert_allclose(all_ranks.mean(0), list(result['mean_ranks'].values()), atol=1e-14, rtol=0)
    for record in result['tiny']:
        task = record['task']; ids, names, values = tables[task]
        values = values[:, [names.index(a) for a in record['pool']]]
        keep = np.isfinite(values).all(axis=1)
        coverage = result['tiny_coverage'][task]
        assert coverage['excluded'] == [name for name, retain in zip(ids, keep) if not retain]
        assert coverage['complete'] == int(keep.sum()) and coverage['released'] == len(ids)
        ids = [name for name, retain in zip(ids, keep) if retain]
        values = values[keep]
        for label, key in [('random', 'first_indices'), ('selected', 'indices')]:
            selected = record['selection'][key]
            assert len(selected) == len(set(selected))
            for pool in ('seen', 'unseen'):
                r = ranks(values[:, record[pool]], task)
                error = np.abs(r[selected].mean(0) - r.mean(0)).mean()
                np.testing.assert_allclose(error, record[label][pool + '_mae'], atol=1e-14, rtol=0)
        assert record['selected_ids'] == [ids[i] for i in record['selection']['indices']]
    out = dict(status='PASS', artifact=str(path.relative_to(ROOT)),
               complete_tasks=len(all_ranks), subset_comparisons=len(result['tiny']),
               scope='Independent raw-cell parsing and pairwise average ranks; all selected/random seen/unseen MAEs and selected dataset identities; not selection optimization or raw-data validity')
    print(json.dumps(out))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    args = parser.parse_args()
    rows = [check(path.resolve()) for path in args.artifacts]
    (ROOT.parent / 'reviews/lesson-quality-audit-047-070/058-ranks.json').write_text(
        json.dumps(dict(status='PASS', records=rows), indent=2) + '\n')

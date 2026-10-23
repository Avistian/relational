"""Read-only verifier probe; disposable notebook mutations only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import warnings

import nbformat

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('review_notebooks', root / 'tests/notebooks/check_notebooks.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

notebooks = sorted(root.rglob('*.ipynb'))
before = {str(p.relative_to(root)): digest(p) for p in notebooks}
with tempfile.TemporaryDirectory(prefix='rel08-verifier-') as directory:
    tree = Path(directory)
    for folder in ('labs', 'labs/html', 'labs/solutions'):
        (tree / folder).mkdir(parents=True, exist_ok=True)
    name = '0081-mpnn-framework'
    clean = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': []}
    for folder in ('labs', 'labs/solutions'):
        (tree / folder / (name + '.ipynb')).write_text(json.dumps(clean))
    (tree / 'labs/html' / (name + '.html')).write_text('<html></html>')
    assert checker.audit(tree)['errors'] == []
    solution = tree / 'labs/solutions' / (name + '.ipynb')
    solution.unlink()
    report = checker.audit(tree)
    assert report['errors'] == ['student_without_solution: labs/0081-mpnn-framework.ipynb']
    print('PASS: deleting a currently shipped solution is rejected:', report['errors'])
    solution.write_text(json.dumps(clean))
    cell = {'cell_type': 'markdown', 'metadata': {}, 'id': 'duplicate', 'source': 'Example'}
    malformed = dict(clean, cells=[copy.deepcopy(cell), copy.deepcopy(cell)])
    solution.write_text(json.dumps(malformed))
    report = checker.audit(tree)
    categories = [w['category'] for w in report['warnings']]
    print('DUPLICATE-ID AUDIT:', json.dumps({'errors': report['errors'], 'warning_categories': categories, 'count': report['counts']['notebooks']}))
    assert report['errors'] == [], 'Probe expected current checker to accept duplicate IDs'
    assert 'DuplicateCellId' in categories
    assert json.loads(solution.read_text())['cells'][1]['id'] == 'duplicate'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            nbformat.validate(copy.deepcopy(malformed), repair_duplicate_cell_ids=False)
        except nbformat.ValidationError as error:
            print('STRICT LIBRARY VALIDATION REJECTS:', str(error).splitlines()[0])
        else:
            raise AssertionError('Expected strict library validation to reject duplicate IDs')
assert before == {str(p.relative_to(root)): digest(p) for p in notebooks}
print('PASS: all 410 repository notebook hashes unchanged')
print('FINDING: malformed duplicate-ID notebook returns zero audit errors; emitted warning is not a blocking result')

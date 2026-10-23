"""Real checker CLI on disposable paired notebooks; never edits course content."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='rel08-r1-cli-') as directory:
    tree = Path(directory)
    for folder in ('tests/notebooks', 'labs', 'labs/html', 'labs/solutions'):
        (tree / folder).mkdir(parents=True, exist_ok=True)
    checker = tree / 'tests/notebooks/check_notebooks.py'
    checker.write_bytes((root / 'tests/notebooks/check_notebooks.py').read_bytes())
    clean = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': []}
    student = tree / 'labs/0081-mpnn-framework.ipynb'
    solution = tree / 'labs/solutions/0081-mpnn-framework.ipynb'
    student.write_text(json.dumps(clean))
    (tree / 'labs/html/0081-mpnn-framework.html').write_text('<html></html>')
    cell = {'cell_type': 'markdown', 'metadata': {}, 'id': 'duplicate', 'source': 'Example'}
    for case, cells, expected in (
        ('duplicate', [dict(cell), dict(cell)], 1),
        ('missing', [{k: v for k, v in cell.items() if k != 'id'}], 0),
    ):
        solution.write_text(json.dumps(dict(clean, cells=cells)))
        before = solution.read_bytes()
        result = subprocess.run([sys.executable, '-B', str(checker)], capture_output=True, text=True)
        report = json.loads(result.stdout)
        print(json.dumps({'case': case, 'exit_code': result.returncode, 'report': report,
                          'stderr': result.stderr, 'bytes_unchanged': solution.read_bytes() == before}, indent=2))
        assert result.returncode == expected
        assert solution.read_bytes() == before
        assert bool(report['errors']) == (case == 'duplicate')
        assert [w['category'] for w in report['warnings']] == [
            'DuplicateCellId' if case == 'duplicate' else 'MissingIDFieldWarning']
    solution.write_text(json.dumps(clean))
    solution.unlink()
    result = subprocess.run([sys.executable, '-B', str(checker)], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report['errors'] == ['student_without_solution: labs/0081-mpnn-framework.ipynb']
    print('PASS: deleting shipped solution still blocks:', report['errors'])
print('PASS: duplicate CLI exit 1; missing-ID CLI exit 0; disposable notebook bytes unchanged')

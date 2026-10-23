"""Cell identity policy through the real audit and disposable complete pairs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class CellIdentityChecks(unittest.TestCase):
    def audit_fixture(self, cells):
        spec = importlib.util.spec_from_file_location(
            'cell_identity_checker', Path(__file__).with_name('check_notebooks.py'))
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory(prefix='rel08-cell-ids-') as directory:
            root = Path(directory)
            for folder in ('labs', 'labs/html', 'labs/solutions'):
                (root / folder).mkdir(parents=True)
            clean = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': []}
            (root / 'labs/0081-mpnn-framework.ipynb').write_text(json.dumps(clean))
            (root / 'labs/html/0081-mpnn-framework.html').write_text('<html></html>')
            solution = root / 'labs/solutions/0081-mpnn-framework.ipynb'
            solution.write_text(json.dumps(dict(clean, cells=cells)))
            before = solution.read_bytes()
            report = checker.audit(root)
            self.assertEqual(solution.read_bytes(), before, 'Audit must preserve notebook bytes')
            self.assertEqual(report['counts']['notebooks'], 2)
            self.assertTrue(all(not paths for paths in report['pairs'].values()))
            return report

    def test_duplicate_cell_ids_block(self):
        cell = {'cell_type': 'markdown', 'metadata': {}, 'id': 'duplicate', 'source': 'Example'}
        report = self.audit_fixture([dict(cell), dict(cell)])
        self.assertEqual(len(report['errors']), 1, 'Duplicate IDs must block the real audit')
        self.assertIn('labs/solutions/0081-mpnn-framework.ipynb', report['errors'][0])
        self.assertIn("Non-unique cell id 'duplicate'", report['errors'][0])

    def test_missing_cell_ids_remain_nonblocking_warnings(self):
        cell = {'cell_type': 'markdown', 'metadata': {}, 'source': 'Example'}
        report = self.audit_fixture([dict(cell), dict(cell)])
        self.assertEqual(report['errors'], [])
        self.assertEqual([warning['category'] for warning in report['warnings']],
                         ['MissingIDFieldWarning', 'MissingIDFieldWarning'])
        self.assertTrue(all(warning['path'] == 'labs/solutions/0081-mpnn-framework.ipynb'
                            for warning in report['warnings']))

"""Exercise the checker against disposable drift and actual repository notebooks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NotebookChecks(unittest.TestCase):
    def checker(self):
        path = Path(__file__).with_name('check_notebooks.py')
        self.assertTrue(path.is_file(), 'Notebook schema/pair checker is required')
        spec = importlib.util.spec_from_file_location('notebook_check', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, root):
        for directory in ('labs', 'labs/html', 'labs/solutions'):
            (root / directory).mkdir(parents=True, exist_ok=True)
        notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': []}
        for path in ('labs/0001-example.ipynb', 'labs/solutions/0001-example.ipynb'):
            (root / path).write_text(json.dumps(notebook))
        (root / 'labs/html/0001-example.html').write_text('<html>Example</html>')

    def test_complete_pairs_and_counts(self):
        checker = self.checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            report = checker.audit(root)
            self.assertEqual(report['errors'], [])
            self.assertEqual(report['counts'], {'notebooks': 2, 'students': 1, 'html': 1, 'solutions': 1})
            self.assertTrue(all(not paths for paths in report['pairs'].values()))

    def test_schema_invalid_notebook_outside_lab_pairs_blocks(self):
        checker = self.checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            extra = root / 'vendor/upstream.ipynb'
            extra.parent.mkdir()
            extra.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [{'cell_type': 'code', 'metadata': {}, 'id': 'bad', 'source': '', 'outputs': 'invalid', 'execution_count': None}]}))
            report = checker.audit(root)
            self.assertEqual(report['counts']['notebooks'], 3)
            self.assertTrue(any('vendor/upstream.ipynb' in error for error in report['errors']))

    def test_non_json_notebook_blocks(self):
        checker = self.checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'labs/0001-example.ipynb').write_text('{broken')
            self.assertTrue(any('labs/0001-example.ipynb' in error for error in checker.audit(root)['errors']))

    def test_deleted_html_and_solution_are_not_baselined(self):
        checker = self.checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'labs/html/0001-example.html').unlink()
            (root / 'labs/solutions/0001-example.ipynb').unlink()
            report = checker.audit(root)
            self.assertEqual(report['pairs']['student_without_html'], ['labs/0001-example.ipynb'])
            self.assertEqual(report['pairs']['student_without_solution'], ['labs/0001-example.ipynb'])
            self.assertEqual(len(report['errors']), 2)

    def test_orphan_html_and_solution_block(self):
        checker = self.checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'labs/0001-example.ipynb').unlink()
            report = checker.audit(root)
            self.assertEqual(report['pairs']['html_without_student'], ['labs/html/0001-example.html'])
            self.assertEqual(report['pairs']['solution_without_student'], ['labs/solutions/0001-example.ipynb'])
            self.assertEqual(len(report['errors']), 2)

    def test_new_missing_solution_is_not_covered_by_old_ignore_policy(self):
        checker = self.checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'labs/0002-new.ipynb').write_text((root / 'labs/0001-example.ipynb').read_text())
            (root / 'labs/html/0002-new.html').write_text('<html>New</html>')
            self.assertTrue(any('labs/0002-new.ipynb' in error for error in checker.audit(root)['errors']))

    def test_actual_repository_validates_every_notebook_and_reports_pairs(self):
        checker = self.checker()
        report = checker.audit(ROOT)
        print(json.dumps(report, indent=2, sort_keys=True))
        actual = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob('*.ipynb'))
        self.assertEqual(report['validated'], actual)
        self.assertEqual(report['counts']['notebooks'], len(actual))
        self.assertGreater(report['counts']['students'], 0)
        self.assertEqual(report['errors'], [])
        for group, paths in report['pairs'].items():
            self.assertEqual(set(paths), set(report['exceptions'][group]))
            self.assertTrue(all(report['exceptions'][group].values()))

"""Read-only manifest checks against real assets and disposable JSON mutations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def check(root, manifest):
    path = Path(__file__).with_name('check_manifest.py')
    # A missing implementation produces an assertion, never an import error in RED.
    assert path.is_file(), 'Manifest integrity checker is not implemented'
    spec = importlib.util.spec_from_file_location('manifest_integrity', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(root, manifest)


class ManifestIntegrity(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / 'lessons/manifest.json').read_text())

    def mutated_errors(self, manifest):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            path.write_text(json.dumps(manifest))
            return check(ROOT, path)

    def test_real_manifest_references_and_coverage(self):
        self.assertEqual(check(ROOT, ROOT / 'lessons/manifest.json'), [])

    def test_deleted_entry_in_disposable_manifest_is_rejected(self):
        removed = self.manifest['lessons'].pop(0)
        errors = self.mutated_errors(self.manifest)
        self.assertTrue(any(removed['slug'] in e and 'unlisted lesson' in e
                            for e in errors), errors)
        print('Deleted-entry mutation rejected:', *errors, sep='\n')

    def test_missing_entry_file_is_rejected(self):
        self.manifest['lessons'][0]['slug'] = 'missing-manifest-lesson-fixture'
        errors = self.mutated_errors(self.manifest)
        self.assertTrue(any('missing lesson file' in e and
                            'missing-manifest-lesson-fixture' in e for e in errors), errors)

    def test_missing_referenced_notebook_is_rejected(self):
        self.manifest['lessons'][0]['labPath'] = 'labs/missing-manifest-fixture.ipynb'
        errors = self.mutated_errors(self.manifest)
        self.assertTrue(any('missing lab file' in e and
                            'labs/missing-manifest-fixture.ipynb' in e for e in errors), errors)

    def test_missing_referenced_python_lab_is_rejected(self):
        self.manifest['lessons'][0]['labPath'] = 'labs/missing-manifest-fixture.py'
        errors = self.mutated_errors(self.manifest)
        self.assertTrue(any('missing lab file' in e and
                            'labs/missing-manifest-fixture.py' in e for e in errors), errors)

    def test_duplicate_numeric_and_bridge_ids_are_rejected(self):
        for identity in (1, 'B01'):
            with self.subTest(identity=identity):
                manifest = copy.deepcopy(self.manifest)
                entry = next(e for e in manifest['lessons'] if e['id'] == identity)
                manifest['lessons'].append(copy.deepcopy(entry))
                errors = self.mutated_errors(manifest)
                self.assertTrue(any('duplicate id' in e and str(identity) in e
                                    for e in errors), errors)

    def test_authored_range_gap_is_rejected_even_at_endpoint(self):
        for identity in (178, 200):
            with self.subTest(identity=identity):
                manifest = copy.deepcopy(self.manifest)
                manifest['lessons'] = [e for e in manifest['lessons'] if e['id'] != identity]
                errors = self.mutated_errors(manifest)
                self.assertTrue(any('authored gap' in e and str(identity) in e
                                    for e in errors), errors)

    def test_only_documented_179_gap_is_allowed(self):
        self.assertIn('| 179 | Failure modes — planned, not yet available |',
                      (ROOT / 'CURRICULUM.md').read_text())
        errors = check(ROOT, ROOT / 'lessons/manifest.json')
        self.assertFalse(any('authored gap' in e for e in errors), errors)


if __name__ == '__main__':
    unittest.main()

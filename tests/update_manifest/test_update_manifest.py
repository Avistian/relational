"""CLI contract in disposable course trees; never regenerate course content."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/update-manifest.py'


class UpdateManifest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'lessons').mkdir()
        (self.root / 'labs').mkdir()
        shutil.copyfile(SCRIPT, self.root / 'scripts/update-manifest.py')
        for slug, title in [('0001-first', 'First &amp; next'),
                            ('0010-exam', 'Exam'), ('b01-bridge', 'Bridge')]:
            (self.root / 'lessons' / (slug + '.html')).write_text(
                '<h1>' + title + '</h1>', encoding='utf-8')
        (self.root / 'labs/0001-alternate.ipynb').write_text('{}')
        (self.root / 'labs/b01-bridge.ipynb').write_text('{}')
        self.manifest = self.root / 'lessons/manifest.json'
        self.expected = {'version': 1, 'lessons': [
            {'id': 1, 'slug': '0001-first', 'year': 1, 'quarter': 1,
             'checkpoint': True, 'labPath': 'labs/0001-alternate.ipynb',
             'title': 'First & next', 'published': False},
            {'id': 10, 'slug': '0010-exam', 'year': 1, 'quarter': 1,
             'checkpoint': True, 'labPath': None, 'title': 'Exam', 'published': True},
            {'id': 'B01', 'slug': 'b01-bridge', 'year': 5, 'quarter': 5,
             'sortOrder': 201.25, 'checkpoint': False,
             'labPath': 'labs/b01-bridge.ipynb', 'title': 'Bridge', 'published': True}]}
        self.write(self.expected)

    def write(self, value):
        self.manifest.write_text(json.dumps(value), encoding='utf-8')

    def snapshot(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob('*') if p.is_file()}

    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-B',
                               str(self.root / 'scripts/update-manifest.py'), *args],
                              capture_output=True, text=True, cwd=self.root)

    def checked(self, succeeds):
        before = self.snapshot()
        result = self.run_cli('--check')
        self.assertEqual(self.snapshot(), before, 'check must not write any file')
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('manifest', (result.stdout + result.stderr).lower())
        self.assertNotIn('Traceback', result.stderr)

    def test_check_full_parsed_content_order_metadata_and_no_writes(self):
        # Formatting and object-key order do not affect parsed equality.
        self.manifest.write_text(json.dumps(self.expected, indent=4, sort_keys=True))
        self.checked(True)
        mutations = []
        for field, value in [('title', 'Stale'), ('year', 9), ('quarter', 9),
                             ('labPath', None), ('slug', 'wrong'), ('id', 99)]:
            changed = copy.deepcopy(self.expected)
            changed['lessons'][0][field] = value
            mutations.append((field, changed))
        for label in ['order', 'missing-entry', 'extra-entry', 'version', 'extra-metadata']:
            changed = copy.deepcopy(self.expected)
            if label == 'order':
                changed['lessons'].reverse()
            elif label == 'missing-entry':
                changed['lessons'].pop()
            elif label == 'extra-entry':
                changed['lessons'].append(copy.deepcopy(changed['lessons'][0]))
            elif label == 'version':
                changed['version'] = 2
            else:
                changed['unexpected'] = 'metadata'
            mutations.append((label, changed))
        for label, changed in mutations:
            with self.subTest(label=label):
                self.write(changed)
                self.checked(False)
        for label, raw in [('missing', None), ('invalid-json', '{'),
                           ('invalid-shape', '[]'), ('invalid-lessons', '{"lessons": null}'),
                           ('invalid-entry', '{"lessons": [null]}')]:
            with self.subTest(label=label):
                if raw is None:
                    self.manifest.unlink()
                else:
                    self.manifest.write_text(raw)
                self.checked(False)
        self.write(self.expected)
        (self.root / 'lessons/0001-first.html').write_text('<h1>Changed source</h1>')
        self.checked(False)

    def test_generation_preserves_existing_contract(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.manifest.read_text(), json.dumps(self.expected, indent=2) + '\n')
        self.assertIn('Wrote 3 entries', result.stdout)
        self.manifest.unlink()
        (self.root / 'lessons/b01-bridge.html').unlink()
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        generated = json.loads(self.manifest.read_text())
        self.assertTrue(generated['lessons'][0]['published'])
        self.assertFalse(generated['lessons'][0]['checkpoint'])
        self.assertEqual(len(generated['lessons']), 2)


if __name__ == '__main__':
    unittest.main()

"""Public CLI contract. Fixtures are synthetic; no test-local implementation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/pages_publish_list.py'


class PublishList(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for directory in ('lessons', 'labs/tree/nested', 'labs/tree/.hidden', '.github/workflows'):
            (self.root / directory).mkdir(parents=True)
        for name in ('index.html', 'labs/tree/a.txt', 'labs/tree/nested/b.json',
                     'labs/tree/.hidden/c.txt', 'labs/a.ipynb', 'labs/omitted.txt',
                     'lessons/0001-present.html'):
            (self.root / name).write_text('fixture')
        (self.root / 'lessons/manifest.json').write_text(json.dumps({'lessons': [
            {'id': 1, 'slug': '0001-present', 'labPath': 'labs/omitted.txt'},
            {'id': 2, 'slug': '0002-missing', 'labPath': 'labs/missing.ipynb'}]}))
        self.plan = self.root / '.github/workflows/pages.yml'
        self.plan.write_text('jobs:\n  build:\n    steps:\n      - name: Build site\n        run: |\n'
                             '          mkdir -p public/labs public/contents\n'
                             '          cp index.html public/\n'
                             '          cp -r lessons public/\n'
                             '          cp -r labs/tree/. public/contents/\n'
                             '          cp labs/*.ipynb public/labs/ 2>/dev/null || true\n')

    def test_recursive_glob_dotfiles_and_stable_deduped_mapping(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                 '--workflow', str(self.plan)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['files'], ['contents/.hidden/c.txt', 'contents/a.txt',
                         'contents/nested/b.json', 'index.html', 'labs/a.ipynb',
                         'lessons/0001-present.html', 'lessons/manifest.json'])
        self.assertEqual(report['sources']['contents/a.txt'], 'labs/tree/a.txt')
        again = subprocess.run(result.args, capture_output=True, text=True)
        self.assertEqual(again.stdout, result.stdout)

    def test_missing_inputs_and_manifest_omissions_are_separate(self):
        with self.plan.open('a') as stream:
            stream.write('          cp labs/absent*.json labs/missing.txt public/labs/\n')
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                 '--workflow', str(self.plan)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['missing_sources'], ['labs/absent*.json', 'labs/missing.txt'])
        self.assertEqual(report['missing_references'], ['labs/missing.ipynb', 'lessons/0002-missing.html'])
        self.assertEqual(report['omitted_references'], ['labs/omitted.txt'])
        self.assertNotIn('labs/omitted.txt', report['files'])

    def test_current_loop_condition_and_removal_semantics(self):
        for number in range(58, 71):
            (self.root / f'labs/_verify_l{number:03d}_results.json').write_text('{}')
            (self.root / f'labs/l{number:03d}-reproduction.md').write_text('x')
        with self.plan.open('a') as stream:
            stream.write('          if [ -f labs/missing.json ]; then cp labs/missing.json public/labs/; fi\n'
                         '          if [ -f index.html ]; then cp index.html public/labs/; fi\n'
                         '          for N in $(seq 58 70); do\n'
                         '            PAD=$(printf \'%03d\' "$N")\n'
                         '            cp "labs/_verify_l${PAD}_results.json" "labs/l${PAD}-reproduction.md" public/labs/\n'
                         '          done\n'
                         '          rm -rf public/contents/nested\n')
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                 '--workflow', str(self.plan)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(len([p for p in report['files'] if p.startswith('labs/')]), 28)
        self.assertNotIn('contents/nested/b.json', report['files'])
        self.assertEqual(report['missing_sources'], [])

    def test_compare_equality_and_difference_have_truthful_exit_codes(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                 '--workflow', str(self.plan), '--compare', str(self.plan)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['comparison'], {'equal': True, 'only_generated': [], 'only_workflow': []})
        other = self.root / 'other.yml'
        other.write_text(self.plan.read_text() + '          cp labs/omitted.txt public/labs/\n')
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                 '--workflow', str(self.plan), '--compare', str(other)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)['comparison']['only_workflow'], ['labs/omitted.txt'])

    def test_unsupported_shell_fails_closed_without_execution(self):
        with self.plan.open('a') as stream:
            stream.write('          cp $(touch SHOULD_NOT_EXIST) public/\n')
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                 '--workflow', str(self.plan)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('unsupported', result.stderr.lower())
        self.assertFalse((self.root / 'SHOULD_NOT_EXIST').exists())

    def test_destination_and_source_escapes_fail_closed(self):
        for command in ('cp index.html ../escape/', 'cp ../outside public/',
                        'cp /etc/passwd public/', 'rm -rf public/../labs'):
            self.plan.write_text('jobs:\n  build:\n    steps:\n      - name: Build site\n        run: |\n          ' + command + '\n')
            result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                     '--workflow', str(self.plan)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, (command, result.stderr))

    def test_default_plan_matches_fixture_subset_without_adding_manifest_references(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertIn('index.html', report['files'])
        self.assertNotIn('labs/omitted.txt', report['files'])


if __name__ == '__main__':
    unittest.main()

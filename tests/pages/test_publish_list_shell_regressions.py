"""Regression fixtures for the verifier's real shell file-set disagreements."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/pages_publish_list.py'


class ShellRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='rel16-shell-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('a.txt', 'b.txt'):
            (self.root / name).write_text('fixture')
        self.base = self.workflow('base.yml', 'cp *.txt public/ 2>/dev/null || true')

    def workflow(self, name, plan):
        path = self.root / name
        path.write_text('jobs:\n  build:\n    steps:\n      - name: Build site\n        run: |\n' +
                        ''.join('          ' + line + '\n' for line in plan.splitlines()))
        return path

    def shell_files(self, plan):
        public = self.root / 'public'
        if public.exists():
            shutil.rmtree(public)
        public.mkdir()
        result = subprocess.run(['bash', '-eu', '-c', plan], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return sorted(path.relative_to(public).as_posix()
                      for path in public.rglob('*') if path.is_file())

    def assert_rejected(self, other):
        # Reject the unsupported plan in either CLI position, without emitting
        # a success-shaped JSON report. Generation must fail closed too.
        for args in (('--workflow', other),
                     ('--workflow', self.base, '--compare', other),
                     ('--workflow', other, '--compare', self.base)):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root),
                                         *map(str, args)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, (result.stdout, result.stderr))
                self.assertEqual(result.stdout, '')
                self.assertIn('unsupported', result.stderr.lower())

    def test_quoted_wildcards_fail_closed_instead_of_comparing_equal(self):
        self.assertEqual(self.shell_files('cp *.txt public/'), ['a.txt', 'b.txt'])
        for operand in ('"*.txt"', "'*.txt'", '"*".txt', "'*'.txt",
                        '"?.txt"', '"[ab].txt"', '"*.txt"""'):
            with self.subTest(operand=operand):
                plan = f'cp {operand} public/ 2>/dev/null || true'
                self.assertEqual(self.shell_files(plan), [])
                other = self.workflow('other.yml', plan)
                self.assert_rejected(other)

    def test_wildcard_removals_fail_closed_instead_of_comparing_equal(self):
        self.assertEqual(self.shell_files('cp *.txt public/'), ['a.txt', 'b.txt'])
        for operand in ('public/*.txt', 'public/?.txt', 'public/[ab].txt', 'public/*'):
            with self.subTest(operand=operand):
                plan = f'cp *.txt public/\nrm -rf {operand}'
                self.assertEqual(self.shell_files(plan), [])
                other = self.workflow('other.yml', plan)
                self.assert_rejected(other)

    def test_public_root_removals_fail_closed_instead_of_comparing_equal(self):
        self.assertEqual(self.shell_files('cp *.txt public/'), ['a.txt', 'b.txt'])
        for operand in ('public', 'public/', 'public//', './public//'):
            with self.subTest(operand=operand):
                plan = f'cp *.txt public/\nrm -rf {operand}'
                self.assertEqual(self.shell_files(plan), [])
                other = self.workflow('other.yml', plan)
                self.assert_rejected(other)


if __name__ == '__main__':
    unittest.main()

"""Exercise the public CLI against a disposable published site."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/check_links.py'


class PublishedLinks(unittest.TestCase):
    def test_cli_reports_published_local_targets_without_changing_site(self):
        self.assertTrue(SCRIPT.is_file(), 'Published HTML checker CLI must exist')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'site'
            root.mkdir()

            def write(name, text):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')

            write('index.html', '\n'.join([
                '<a href="lessons/topic.html?view=1#A%20B">decoded</a>',
                '<a href="lessons/topic.html#literal%20id">literal</a>',
                '<a href="lessons/topic.html#literal%2520id">double encoded</a>',
                '<a href="lessons/topic.html#old">named</a>',
                '<a href="lessons/topic.html#:~:text=hello">text</a>',
                '<a href="lessons/topic.html#top">top</a>',
                '<a href="reference/">directory</a>',
                '<img src="assets/a%20b.svg"/>',
                '<video poster="assets/a%20b.svg"></video>',
                '<a href="file.pdf#page=2">non HTML</a>',
                '<a href="https://example.invalid/no">external</a>',
                '<a href="//example.invalid/no">external</a>',
                '<a href="mailto:a@example.invalid">mail</a>',
                '<a href="vendor/missing.html">vendor</a>',
                '<a href="archive/missing.html">archive</a>',
                '<a href="missing.html">missing file</a>',
                '<a href="lessons/topic.html#a%20b">case sensitive</a>',
                '<a href="lessons/topic.html#absent">missing fragment</a>',
                '<a href="escape.html">outside symlink</a>',
            ]))
            write('lessons/topic.html', '<h1 id="A B">A</h1><p id="literal%20id">B</p><a name="old"></a><a href="/index.html">home</a>')
            write('labs/html/exercise.html', '<h2 id="here">Lab</h2><a href="#here">self</a>')
            write('reference/index.html', '')
            write('lessons/.published/page.html', '')
            write('assets/a b.svg', '<svg/>')
            write('file.pdf', 'fixture')
            for name in ('lessons/vendor/bad.html', 'reference/archive/bad.html',
                         'labs/sources/bad.html', 'public-test/bad.html'):
                write(name, '<a href="missing.html">outside source scope</a>')
            outside = Path(directory) / 'outside.html'
            outside.write_text('<a href="bad.html">not a source</a>')
            (root / 'escape.html').symlink_to(outside)

            def snapshot():
                return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob('*') if p.is_file()}

            before = snapshot()
            command = [sys.executable, '-B', str(SCRIPT), '--root', str(root)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=30)
            second = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 1, first.stderr)
            self.assertEqual((first.stdout, first.stderr), (second.stdout, second.stderr))
            report = json.loads(first.stdout)
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(report['pages'], ['index.html', 'labs/html/exercise.html',
                             'lessons/.published/page.html', 'lessons/topic.html', 'reference/index.html'])
            self.assertEqual(report['counts'], {'pages': 5, 'checked': 16, 'broken': 4, 'excluded': 5})
            self.assertEqual([(r['line'], r['url'], r['reason']) for r in report['broken']], [
                (16, 'missing.html', 'missing-file'),
                (17, 'lessons/topic.html#a%20b', 'missing-fragment'),
                (18, 'lessons/topic.html#absent', 'missing-fragment'),
                (19, 'escape.html', 'outside-root')])
            self.assertTrue(all(row in report['checked'] for row in report['broken']))
            self.assertIn('LEAD_RESULT: FAILED', first.stderr)
            self.assertEqual(snapshot(), before)

            # A repaired disposable fixture distinguishes an all-good site from
            # a broken inventory. The checker itself never repairs content.
            write('missing.html', '')
            write('lessons/topic.html', '<p id="A B"></p><p id="literal%20id"></p><a name="old"></a><p id="a b"></p><p id="absent"></p><a href="/index.html">home</a>')
            (root / 'escape.html').unlink()
            write('escape.html', '')
            good = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(good.stdout)['broken'], [])
            self.assertIn('LEAD_RESULT: OK', good.stderr)

            invalid = subprocess.run(command[:-1] + [str(root / 'absent')],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn('error', json.loads(invalid.stdout))


if __name__ == '__main__':
    unittest.main()

"""Stdlib-only CLI/output fixtures: execute no imports, servers or browsers."""
import argparse
import ast
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    'course_fallbacks': ('course-visual-quality-2026-10-07', ''),
    'course_visuals': ('course-visual-quality-2026-10-07', 'after'),
    'lesson_visuals': ('lesson-visuals-2026-10-07', ''),
    'visual_details': ('lesson-visual-refinement-2026-10-07', 'local'),
}


class OutputPathsTests(unittest.TestCase):
    def resolve(self, name, argv):
        # Evaluate only the actual argparse and output assignments from main's
        # pre-browser prefix. This also works on the original top-level fallback.
        tree = ast.parse((ROOT / f'scripts/browser_check_{name}.py').read_text())
        main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
        statements = main.body if main else tree.body
        prefix = []
        for node in statements:
            if isinstance(node, ast.ClassDef):
                if main:
                    break
                continue
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id in {'parser', 'ap', 'args', 'out', 'report_dir'} for t in node.targets):
                    prefix.append(node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument':
                    prefix.append(node)
        namespace = {'argparse': argparse, 'Path': Path, 'ROOT': ROOT, 'root': ROOT}
        with patch.object(sys, 'argv', ['browser-check', *argv]), contextlib.redirect_stderr(io.StringIO()):
            try:
                exec(compile(ast.Module(body=prefix, type_ignores=[]), '<cli-prefix>', 'exec'), namespace)
            except SystemExit as exc:
                self.fail(f'{name} rejected CLI {argv}: exit {exc.code}')
        return namespace.get('out', namespace.get('report_dir'))

    def test_out_selects_review_directory_for_all_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'review results'
            for name in SCRIPTS:
                with self.subTest(script=name):
                    self.assertEqual(self.resolve(name, ['--out', str(out)]), out)
            self.assertFalse(out.exists(), 'parser/output resolution must be read-only')

    def test_out_wins_over_report_dir_in_either_order(self):
        for name in SCRIPTS:
            for argv in (['--out', 'chosen', '--report-dir', 'legacy'], ['--report-dir', 'legacy', '--out', 'chosen']):
                with self.subTest(script=name, argv=argv):
                    self.assertEqual(self.resolve(name, argv), Path('chosen'))

    def test_defaults_preserve_existing_destinations(self):
        for name, (directory, suffix) in SCRIPTS.items():
            with self.subTest(script=name):
                self.assertEqual(self.resolve(name, []), ROOT / 'reviews' / directory / suffix)

    def test_legacy_report_dir_preserves_suffixes(self):
        for name in ('lesson_visuals', 'visual_details'):
            with self.subTest(script=name):
                self.assertEqual(self.resolve(name, ['--report-dir', 'legacy']), Path('legacy') / SCRIPTS[name][1])

    def test_live_defaults_and_out(self):
        for name in ('course_visuals', 'lesson_visuals', 'visual_details'):
            with self.subTest(script=name):
                directory, _ = SCRIPTS[name]
                suffix = '' if name == 'lesson_visuals' else 'live'
                self.assertEqual(self.resolve(name, ['--base-url', 'https://example.invalid']), ROOT / 'reviews' / directory / suffix)
                self.assertEqual(self.resolve(name, ['--base-url', 'https://example.invalid', '--out', 'chosen']), Path('chosen'))

    def test_review_literals_only_occur_in_argument_defaults_and_export_is_preserved(self):
        for name in SCRIPTS:
            tree = ast.parse((ROOT / f'scripts/browser_check_{name}.py').read_text())
            defaults = {id(n) for call in ast.walk(tree) if isinstance(call, ast.Call)
                        for kw in call.keywords if kw.arg == 'default' for n in ast.walk(kw.value)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'reviews/' in node.value:
                    with self.subTest(script=name):
                        self.assertIn(id(node), defaults)
        tree = ast.parse((ROOT / 'scripts/browser_check_course_visuals.py').read_text())
        dest = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'dest' for t in n.targets))
        self.assertEqual(eval(compile(ast.Expression(dest), '<export-path>', 'eval'), {'ROOT': ROOT}), ROOT / 'assets/course-visuals')

    def test_generated_directories_ignored_and_tracked_pngs_retained(self):
        for path in ('reviews/rel15-fixture/new.png', '.scratch/rel15-fixture/new.json'):
            result = subprocess.run(['git', 'check-ignore', '--no-index', path], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr or path)
        baseline = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '503704f', '--', 'reviews'], cwd=ROOT, text=True).splitlines()
        tracked = set(subprocess.check_output(['git', 'ls-files', 'reviews'], cwd=ROOT, text=True).splitlines())
        self.assertTrue(set(baseline).issubset(tracked))
        self.assertTrue(all((ROOT / path).is_file() for path in baseline if path.endswith('.png')))

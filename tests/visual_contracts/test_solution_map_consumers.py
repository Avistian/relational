"""Removing an unused route preserves generator and publication consumers."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'labs'))
import _build_solution_maps as builder


class WriteBoundary(Exception):
    """Stop the real consumer before its first filesystem write."""


@contextmanager
def first_text_output():
    outputs = []
    original_open = Path.open

    def read_only_open(path, mode='r', *args, **kwargs):
        if any(flag in mode for flag in 'wax+'):
            raise AssertionError(f'Unexpected write blocked: {path}')
        return original_open(path, mode, *args, **kwargs)

    def stop_write(path, value, *args, **kwargs):
        outputs.append((path, value))
        raise WriteBoundary

    with patch.object(Path, 'open', read_only_open), \
            patch.object(Path, 'write_text', stop_write), \
            patch.object(Path, 'write_bytes', side_effect=AssertionError('Binary write blocked')):
        yield outputs


class SolutionMapConsumerContracts(unittest.TestCase):
    def test_lesson_58_revision_reaches_html_output_without_exporting_a_map(self):
        with first_text_output() as outputs:
            with self.assertRaises(WriteBoundary):
                builder.revise(58)
        path, html = outputs[0]
        self.assertEqual(path, builder.lesson(58))
        soup = BeautifulSoup(html, 'html.parser')
        self.assertIsNotNone(soup.select_one('[data-solution-story="opening"]'))
        self.assertIsNotNone(soup.select_one('[data-solution-story="handoff"]'))
        self.assertEqual(soup.select('[data-solution-map]'), [])

    def test_atlas_includes_lesson_58_without_an_unused_map_link(self):
        with first_text_output() as outputs:
            with self.assertRaises(WriteBoundary):
                builder.atlas()
        path, html = outputs[0]
        self.assertEqual(path, ROOT / 'reference/solution-map-atlas.html')
        soup = BeautifulSoup(html, 'html.parser')
        self.assertEqual(len(soup.select('article')), 23)
        lesson = soup.select_one('#lesson-58')
        self.assertIsNotNone(lesson)
        self.assertEqual(lesson.h3.a['href'], '../lessons/' + builder.lesson(58).name)
        self.assertEqual(lesson.select('.map-link'), [])
        self.assertNotIn('solution-evidence-funnel', html)

    def test_read_only_architecture_export_is_current(self):
        self.assert_check_succeeds('export_architecture_routes.py')

    def test_read_only_publication_refresh_is_current(self):
        self.assert_check_succeeds('refresh_lesson_visuals.py')

    def assert_check_succeeds(self, script):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / script), '--check'],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

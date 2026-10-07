"""Content/accessibility contracts for the whole-course visual revision."""
from pathlib import Path
import re
import unittest
from bs4 import BeautifulSoup
ROOT = Path(__file__).resolve().parents[1]

def lessons():
    return [p for p in sorted((ROOT/'lessons').glob('*.html')) if re.match(r'b\d|\d{4}', p.name)]

class VisualContracts(unittest.TestCase):
    def test_every_lesson_has_static_visual_navigation(self):
        for path in lessons():
            with self.subTest(lesson=path.name):
                soup = BeautifulSoup(path.read_text(), 'html.parser')
                self.assertIsNotNone(soup.select_one('[data-visual-reading]'))
                for a in soup.select('[data-visual-reading] a[href^="#"]'):
                    self.assertIsNotNone(soup.find(id=a['href'][1:]))

    def test_visual_stories_have_sources_and_accessible_fallbacks(self):
        found = 0
        for path in lessons():
            soup = BeautifulSoup(path.read_text(), 'html.parser')
            for story in soup.select('.visual-story'):
                found += 1
                self.assertIsNotNone(story.select_one('svg title'))
                self.assertIsNotNone(story.select_one('svg desc'))
                self.assertEqual(len(story.select('[data-visual-step]')), 3)
                self.assertIsNotNone(story.select_one('a[href^="https://"]'))
                self.assertIsNotNone(story.select_one('details summary'))
        self.assertGreaterEqual(found, 40)

    def test_all_visual_assets_exist(self):
        for path in lessons():
            soup = BeautifulSoup(path.read_text(), 'html.parser')
            for element in soup.select('[data-visual-reading] a, .visual-story a, link[href*="visual"], script[src*="visual"]'):
                href = element.get('href', element.get('src', ''))
                if href.startswith(('../assets/', '../reference/')):
                    self.assertTrue((path.parent/href.split('#')[0]).exists(), f'{path}: {href}')

if __name__ == '__main__':
    unittest.main()

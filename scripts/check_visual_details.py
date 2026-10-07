"""Regression checks for supplementary mechanism figures and regeneration."""
import unittest
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from refresh_lesson_visuals import selected, update, key_for
from visual_details import DETAILS, render

class DetailsContract(unittest.TestCase):
    def test_every_detail_is_accessible_and_survives_refresh(self):
        for path in selected():
            key = key_for(path)
            if key not in DETAILS:
                continue
            with self.subTest(lesson=key):
                html, _, _ = update(path)
                soup = BeautifulSoup(html, 'html.parser')
                figure = soup.select_one('.visual-detail')
                self.assertIsNotNone(figure)
                self.assertEqual(len(soup.select('.visual-detail')), 1)
                self.assertIsNotNone(figure.select_one('[tabindex="0"]'))
                self.assertTrue(figure.img['alt'])
                self.assertIsNotNone(figure.select_one('details summary'))
                ids = [e['id'] for e in soup.select('[id]')]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(html, path.read_text(), 'Generated lesson is stale')
                svg = ET.fromstring(render(key))
                ns = {'s': 'http://www.w3.org/2000/svg'}
                self.assertTrue(svg.find('s:title', ns).text)
                self.assertTrue(svg.find('s:desc', ns).text)

if __name__ == '__main__':
    unittest.main()

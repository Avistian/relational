"""Authored solution routes must have a real lesson reading route."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'labs'))

from _solution_map_content import MAPS
from check_course_visuals import BeautifulSoup, selected


class SolutionRouteContracts(unittest.TestCase):
    def test_authored_solution_routes_match_rendered_reading_routes(self):
        # Exercise the same real HTML and responsive-map contract as the course
        # checker, using the current authored definitions rather than its export.
        rendered = set()
        for path in selected():
            soup = BeautifulSoup(path.read_text(), 'html.parser')
            rendered.update(
                figure['data-responsive-map']
                for figure in soup.select('figure[data-responsive-map]')
                if figure['data-responsive-map'].startswith('solution-')
            )
        authored = {'solution-' + diagram.key
                    for diagrams in MAPS.values() for diagram in diagrams}
        self.assertEqual(authored, rendered)


if __name__ == '__main__':
    unittest.main()

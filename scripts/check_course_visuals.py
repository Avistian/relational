"""Check that narrow architecture routes preserve authored graph topology."""
import unittest
from bs4 import BeautifulSoup
from refresh_lesson_visuals import selected,key_for
from responsive_architecture import DATA,verify
from course_visual_specs import SPECS
from course_notebooks import verify_exports
class CourseVisualContracts(unittest.TestCase):
 def test_export_sources(self):
  verify();verify_exports()
 def test_authored_graphs_and_reading_routes(self):
  seen=set();traces=set()
  for path in selected():
   soup=BeautifulSoup(path.read_text(),'html.parser');key=key_for(path)
   for figure in soup.select('figure[data-responsive-map]'):
    route=figure['data-responsive-map'];spec=DATA['routes'][route];seen.add(route)
    nodes=figure.select('.rm-nodes > li');self.assertEqual(len(nodes),len(spec['nodes']))
    for actual,expected in zip(nodes,spec['nodes']):
     self.assertEqual(actual['id'],'rm-'+route+'-'+expected['id'])
     self.assertEqual(actual.h4.get_text(),expected['title']);self.assertEqual(actual.find('p').get_text(),expected['body'])
     links=actual.select('a');edges=[e for e in spec['edges'] if e['a']==expected['id']]
     self.assertEqual([a['href'] for a in links],['#rm-'+route+'-'+e['b'] for e in edges])
   cv=soup.select_one('.course-visual')
   if cv:
    traces.add(key);self.assertEqual(len(cv.select('.cv-route > li')),4)
    self.assertIn(SPECS[key]['answer'],cv.select_one('svg title').get_text())
    self.assertEqual(cv.select_one('summary').get_text(),SPECS[key]['question'])
  self.assertEqual(seen,set(DATA['routes']));self.assertEqual(traces,set(SPECS))
if __name__=='__main__':unittest.main()

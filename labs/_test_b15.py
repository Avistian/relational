"""Behavioral specifications: information restrictions and exact finite oracles."""
import math, unittest
from relkit.labels_b15 import visible_labels,predict_rules,predict_columns,mutual_information
class Contracts(unittest.TestCase):
 def test_visibility(self):
  rows=[dict(key=('A',1),available=3,label=0),dict(key=('A',5),available=5,label=1),dict(key=('B',2),available=6,label=1),dict(key=('C',1),available=2,label=1)]
  allowed={r['key'] for r in rows[:3]}
  self.assertEqual(visible_labels(rows,('A',5),5,allowed),[(('A',1),0)])
  rows[1]['label']=0;rows[2]['label']=0;rows[3]['label']=0
  self.assertEqual(visible_labels(rows,('A',5),5,allowed),[(('A',1),0)])
  self.assertEqual(visible_labels(rows,('A',5),3,allowed),[(('A',1),0)])
  with self.assertRaises(ValueError):visible_labels(rows+rows[:1],('A',5),5,allowed)
  with self.assertRaises(ValueError):visible_labels(rows,('A',5),float('nan'),allowed)
 def test_rules(self):
  for bit in (0,1):
   self.assertEqual(predict_rules(bit,[]),.5)
   for theta in (0,1):self.assertEqual(predict_rules(bit,[(0,theta)]),bit^theta)
  with self.assertRaises(ValueError):predict_rules(0,[(0,0),(0,1)])
 def test_columns(self):
  self.assertEqual(predict_columns((0,1),[((0,0),0),((1,1),1)]),.5)
  self.assertEqual(predict_columns((0,1),[((0,1),1)]),1)
  self.assertEqual(predict_columns((1,0),[((0,1),1)]),0)
  with self.assertRaises(ValueError):predict_columns((0,1),[((0,0),1)])
 def test_information(self):
  self.assertAlmostEqual(mutual_information([(0,0),(0,1),(1,0),(1,1)]),0)
  self.assertAlmostEqual(mutual_information([(0,0),(1,1)]),1)
  with self.assertRaises(ValueError):mutual_information([])
if __name__=='__main__':unittest.main()

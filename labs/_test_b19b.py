"""Behavior checks for information access, complete horizons and metric denominators."""
import unittest,math
import numpy as np
from relkit.forecast_b19b import available_features,rolling_origins,forecast_scores
class ForecastTests(unittest.TestCase):
 def test_availability(self):
  rows=[dict(id='calendar',time=12,available_at=0,value=2),dict(id='weather',time=12,available_at=12,value=9),dict(id='late',time=8,available_at=11,value=3)]
  self.assertEqual(available_features(rows,10),{'calendar':2})
  changed=[dict(x,value=999) if x['id']!='calendar' else x for x in rows]
  self.assertEqual(available_features(rows,10),available_features(changed,10))
  self.assertEqual(available_features(rows,12),{'calendar':2,'weather':9,'late':3})
  with self.assertRaises(ValueError):available_features(rows+[rows[0]],10)
  with self.assertRaises(ValueError):available_features([dict(id='a',time=1,value=2)],10)
 def test_origins(self):
  self.assertEqual(rolling_origins(12,4,3,3),[(4,[4,5,6]),(7,[7,8,9])])
  self.assertEqual(rolling_origins(7,4,3,1),[(4,[4,5,6])])
  for args in [(12,4,0,1),(12,4,3,0),(12,0,3,1),(12,11,3,1)]:
   with self.assertRaises(ValueError):rolling_origins(*args)
 def test_scores(self):
  s=forecast_scores([1,3,2,4],[2,4],[[1],[5]],[.5],1)
  self.assertAlmostEqual(s['mase'],.6);self.assertAlmostEqual(s['wql'],1/3)
  self.assertAlmostEqual(s['scale'],5/3)
  # Correct asymmetric quantile loss: q=.1, y=2, prediction=0 -> 2*.1*2/2=.2
  self.assertAlmostEqual(forecast_scores([1,2],[2],[[0,1]],[.1,.5],1)['wql'],.35)
  for h,y,p,q,sn in [([1,1],[2],[[1]],[.5],1),([1,2],[0],[[1]],[.5],1),([1,2],[2],[[3,1]],[.1,.5],1),([1,2],[2],[[1]],[.1],1)]:
   with self.assertRaises(ValueError):forecast_scores(h,y,p,q,sn)
if __name__=='__main__':unittest.main()

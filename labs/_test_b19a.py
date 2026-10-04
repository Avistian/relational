"""Hand-computable behavioral tests, shared with the portable notebook."""
import unittest,math
from relkit.scoring_b19a import crps,interval_score,calibrate
class ScoringTests(unittest.TestCase):
    def test_crps(self):
        self.assertAlmostEqual(crps([-1,1],[.5,.5],0),.5)
        self.assertAlmostEqual(crps([-2,0,2],[.25,.5,.25],0),.25)
        self.assertAlmostEqual(crps([3],[1],-2),5)
        self.assertAlmostEqual(crps([1,-1],[.5,.5],0),.5)
        self.assertAlmostEqual(crps([-2,2],[.5,.5],0),1)
        for v,p,y in [([],[],0),([0],[.9],0),([0,1],[1.1,-.1],0),([0],[1],float('nan')),([float('inf')],[1],0),([0,1],[1],0)]:
            with self.assertRaises(ValueError):crps(v,p,y)
    def test_interval(self):
        self.assertEqual(interval_score(-1,1,2,.2),12)
        self.assertEqual(interval_score(-1,1,-2,.2),12)
        self.assertEqual(interval_score(-1,1,0,.2),2)
        self.assertEqual(interval_score(-1,1,1,.2),2)
        for args in [(1,-1,0,.2),(-1,1,0,0),(-1,1,0,1),(-1,1,float('nan'),.2)]:
            with self.assertRaises(ValueError):interval_score(*args)
    def test_calibration(self):
        rows=[{'id':f'c{i}','split':'calibration','y':i,'mean':0} for i in range(9)]
        r=calibrate(rows,.2)
        self.assertEqual(r,{'n':9,'k':8,'radius':7.0,'ids':[f'c{i}' for i in range(9)]})
        self.assertEqual(calibrate(rows,.01)['radius'],float('inf'))
        for bad in [[],rows+[rows[0]],[dict(rows[0],split='test')],[dict(rows[0],y=float('nan'))]]:
            with self.assertRaises(ValueError):calibrate(bad,.2)
        for alpha in [0,1,float('nan')]:
            with self.assertRaises(ValueError):calibrate(rows,alpha)
if __name__=='__main__':unittest.main()

"""Behavioral oracles: time boundary, unbiasedness and degree-stratified error."""
import itertools,unittest
import numpy as np
from relkit.context_b18 import eligible_history,corrected_total,stratified_metrics

class Boundaries(unittest.TestCase):
    def test_history(self):
        rows=[dict(id='a',day=29,kind='credit',amount=10.,month=1),dict(id='b',day=30,kind='credit',amount=20.,month=1),dict(id='c',day=31,kind='credit',amount=1e9,month=2),dict(id='d',day=20,kind='debit',amount=300.,month=1)]
        self.assertEqual([r['id'] for r in eligible_history(rows,30)],['a','b'])
        rows[2]['amount']=-1e12
        self.assertEqual(sum(r['amount'] for r in eligible_history(rows,30)),30.)
        self.assertEqual(eligible_history(rows,0),[])
        with self.assertRaises(ValueError):eligible_history(rows+[rows[0]],30)
        with self.assertRaises(ValueError):eligible_history([dict(rows[0],amount=float('nan'))],30)
    def test_correction(self):
        values=[1.,2.,7.,10.]
        estimates=[corrected_total(np.array(s),4) for s in itertools.combinations(values,2)]
        self.assertAlmostEqual(sum(estimates)/6,20.)
        self.assertGreater(np.var(estimates),0.)
        self.assertEqual(corrected_total(np.array([]),0),0.)
        self.assertEqual(corrected_total(np.array(values),4),20.)
        with self.assertRaises(ValueError):corrected_total(np.array([]),4)
        with self.assertRaises(ValueError):corrected_total(np.array(values),2)
    def test_metrics(self):
        r=stratified_metrics([8,8,64,64],[1.,3.,2.,4.],[2.,2.,1.,1.])
        self.assertEqual(r['8'],dict(n=2,bias=0.,rmse=1.))
        self.assertAlmostEqual(r['64']['bias'],2.)
        self.assertAlmostEqual(r['64']['rmse'],np.sqrt(5))
        with self.assertRaises(ValueError):stratified_metrics([8],[1,2],[1])
        with self.assertRaises(ValueError):stratified_metrics([8],[float('nan')],[1])

if __name__=='__main__':unittest.main()

"""Behavioral contracts for the learner's load-bearing decisions."""
import unittest
import numpy as np
from relkit.curriculum_b20 import exposure_order,paired_contract,auc_retention

class CurriculumTests(unittest.TestCase):
    def test_order(self):
        levels=np.array([2,0,1,0,2,1]);a=exposure_order(levels,'staged',7,2);b=exposure_order(levels,'shuffled',7,2)
        self.assertEqual(len(a),12)
        np.testing.assert_array_equal(np.bincount(a,minlength=6),np.full(6,2))
        np.testing.assert_array_equal(np.bincount(b,minlength=6),np.full(6,2))
        self.assertTrue(np.all(np.diff(levels[a])>=0))
        np.testing.assert_array_equal(b,exposure_order(levels,'shuffled',7,2))
        self.assertFalse(np.array_equal(a,b))
        with self.assertRaises(ValueError):exposure_order(levels,'best-test',1)
        with self.assertRaises(ValueError):exposure_order([], 'staged',0)
        with self.assertRaises(ValueError):exposure_order(levels,'staged',0,0)
    def test_pair(self):
        a=dict(pool='abc',init='def',updates=12,optimizer={'lr':.001},exposures=[2,2,2],selection='final',evaluation='ghi',batch_size=1)
        self.assertTrue(paired_contract(a,dict(a)))
        for key in a:
            bad=dict(a);bad[key]='changed'
            with self.assertRaises(ValueError,msg=key):paired_contract(a,bad)
        bad=dict(a);del bad['init']
        with self.assertRaises(ValueError):paired_contract(a,bad)
    def test_retention(self):
        got=auc_retention(.638,.725)
        self.assertAlmostEqual(got['raw'],.88)
        self.assertAlmostEqual(got['above_chance'],.138/.225)
        self.assertAlmostEqual(auc_retention(.4,.8)['above_chance'],-1/3)
        for s,b in [(1.1,.8),(.7,.5),(.7,float('nan'))]:
            with self.assertRaises(ValueError):auc_retention(s,b)
if __name__=='__main__':unittest.main()

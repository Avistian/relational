"""Hand-computable behavioral contracts for the three live learner operations."""
import unittest
from relkit.evidence_b19 import split_audit,paired_scores,dataset_summary

class Boundaries(unittest.TestCase):
    def test_split(self):
        rows=[dict(id=i,group=i//2,y=i//2) for i in range(4)]
        self.assertEqual(split_audit(rows,[0,2],[1,3]),dict(train_n=2,test_n=2,overlap_groups=[0,1],group_disjoint=False))
        self.assertTrue(split_audit(rows,[0,1],[2,3])['group_disjoint'])
        for train,test in [([0,0],[1]),([0],[0]),([99],[1]),([],[1])]:
            with self.assertRaises(ValueError):split_audit(rows,train,test)
        with self.assertRaises(ValueError):split_audit(rows+[rows[0]],[0],[1])
    def test_pairs(self):
        rows=[dict(dataset=d,seed=0,model=m,loss=v,origin='missing' if v is None else 'measured') for d,vals in [('x',[.1,.2,.3]),('y',[None,.2,.9])] for m,v in zip(['A','B','RF'],vals)]
        a=paired_scores(rows,['A','B']);self.assertEqual(a['keys'],[['x',0]]);self.assertAlmostEqual(a['means']['A'],.1)
        b=paired_scores(rows,['A','B'],'rf');self.assertAlmostEqual(b['means']['A'],.5);self.assertEqual(b['cells'][-2]['origin'],'imputed:RF');self.assertEqual(b['cells'][-2]['model'],'A')
        for bad in [rows+[rows[0]],[dict(x,loss=float('nan')) if i==0 else x for i,x in enumerate(rows)],[dict(x,origin='measured') if x['loss'] is None else x for x in rows]]:
            with self.assertRaises(ValueError):paired_scores(bad,['A','B'])
        with self.assertRaises(ValueError):paired_scores(rows,['A','B'],'pretend')
        with self.assertRaises(ValueError):paired_scores(rows,['absent','B'],'rf')
    def test_datasets(self):
        rows=[dict(dataset=d,seed=s,delta=v) for d,vals in [('a',[.1,.2]),('b',[-.1,0])] for s,v in enumerate(vals)]
        r=dataset_summary(rows);self.assertEqual(r['n_datasets'],2);self.assertAlmostEqual(r['mean'],.05);self.assertAlmostEqual(r['se_dataset'],.1)
        doubled=rows+[dict(x,seed=x['seed']+2) for x in rows]
        rr=dataset_summary(doubled);self.assertEqual(rr['n_datasets'],2);self.assertAlmostEqual(rr['se_dataset'],r['se_dataset']);self.assertLess(rr['se_naive_rows'],r['se_naive_rows'])
        for bad in [[],rows+[rows[0]],rows+[dict(dataset='c',seed=0,delta=float('nan'))]]:
            with self.assertRaises(ValueError):dataset_summary(bad)
if __name__=='__main__':unittest.main()

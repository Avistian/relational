"""Behavioral specifications using hand-computed oracles and interventions."""
import copy, math, unittest
from relkit.partitions_b16 import block_predict,occupancy,score_subset,select_columns,incidence_partition,run_experiment

def fixture(kind='signal',prefix='tr'):
    return [dict(id=f'{prefix}-{i}',A=i//4,B=(i//2)%2,K=f'{prefix}-{i}',
                 y=i//4 if kind=='signal' else ((i//4)^((i//2)%2) if kind=='xor' else i%2)) for i in range(8)]

class Contracts(unittest.TestCase):
    def test_block_prediction_and_unseen_fallback(self):
        tr=[dict(id='a',A=0,y=0),dict(id='b',A=0,y=1),dict(id='c',A=1,y=1)]
        q=[dict(A=0),dict(A=1),dict(A=2)]
        self.assertEqual(block_predict(tr,q,['A']),[.5,1,2/3])
        self.assertEqual(block_predict(tr,q,[]),[2/3]*3)
        with self.assertRaises(ValueError):block_predict(tr,q,['y'])
        with self.assertRaises(ValueError):block_predict(tr+tr[:1],q,['A'])
    def test_occupancy(self):
        self.assertAlmostEqual(occupancy([8]),1/math.sqrt(8))
        self.assertEqual(occupancy([4,4]),.5)
        self.assertEqual(occupancy([1]*8),1)
        for bad in [[],[0],[2,-1],[float('nan')],[1.5]]:
            with self.assertRaises(ValueError):occupancy(bad)
    def test_key_memorizes_but_fails_on_unseen_rows(self):
        tr,va=fixture(),fixture(prefix='va')
        score=score_subset(tr,va,['K'],.5)
        self.assertEqual(score['risk'],.25)
        self.assertEqual(score['J'],.75)
        self.assertEqual(score_subset(tr,va,['A'],.5)['J'],.25)
        self.assertEqual(score_subset(tr,va,['A'],.5,'0-1')['risk'],0)
        with self.assertRaises(ValueError):score_subset(tr,tr,['A'])
        with self.assertRaises(ValueError):score_subset(tr,va,['A'],float('nan'))
    def test_xor_is_a_search_failure_not_objective_failure(self):
        tr,va=fixture('xor'),fixture('xor','va')
        best=select_columns(tr,va,['A','B','K'],.5)
        fw=select_columns(tr,va,['A','B','K'],.5,'forward')
        bw=select_columns(tr,va,['A','B','K'],.5,'backward')
        self.assertEqual(best['selected'],['A','B'])
        self.assertEqual(fw['selected'],[])
        self.assertEqual(bw['selected'],['A','B'])
        self.assertGreater(fw['J'],best['J'])
        self.assertEqual(select_columns(tr,va,['A','B','K'],1,'backward')['selected'],['A','B'])
        self.assertEqual(select_columns(tr,va,['A','B','K'],1,'exhaustive')['selected'],[])
    def test_abstention_is_available_from_both_directions(self):
        tr,va=fixture('null'),fixture('null','va')
        for mode in ['forward','backward','exhaustive']:
            self.assertEqual(select_columns(tr,va,['A','B','K'],.5,mode)['selected'],[])
    def test_graph_assumptions_are_load_bearing(self):
        rows=fixture()
        self.assertEqual(incidence_partition(rows,['A']),[[0,1,2,3],[4,5,6,7]])
        self.assertEqual(incidence_partition(rows,['A'],False),[list(range(8))])
        self.assertEqual(incidence_partition(rows,['A'],True,['B']),[[0,1],[2,3],[4,5],[6,7]])
        changed=copy.deepcopy(rows)
        for row in changed:row['y']=1-row['y']
        self.assertEqual(incidence_partition(rows,['A']),incidence_partition(changed,['A']))
        with self.assertRaises(ValueError):incidence_partition(rows,['A'],True,['y'])
    def test_complete_grid_and_test_label_interventions(self):
        report=run_experiment()
        self.assertEqual(len(report['subsets']),72)
        self.assertEqual(len(report['selections']),27)
        self.assertEqual(len(report['graph_checks']),24)
        self.assertEqual(len(report['test_interventions']),256)
        self.assertTrue(all(r['unchanged'] for r in report['test_interventions']))

if __name__=='__main__':unittest.main()

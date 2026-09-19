"""Independent examples for constrained validation selection and matched cohorts."""
import unittest
from relkit.decision_guide import select_feasible, matched_datasets

class Decisions(unittest.TestCase):
    def test_budget_changes_choice_not_test_target(self):
        rows=[dict(model='tree',validation_loss=.32,p95_ms=2),dict(model='neural',validation_loss=.29,p95_ms=8)]
        self.assertEqual(select_feasible(rows,5),'tree')
        self.assertEqual(select_feasible(rows,10),'neural')
        rows[0]['test_loss']=0;rows[1]['test_loss']=100
        self.assertEqual(select_feasible(rows,10),'neural')
        self.assertIsNone(select_feasible(rows,1))

    def test_boundary_and_stable_tie(self):
        rows=[dict(model='a',validation_loss=.3,p95_ms=2),dict(model='b',validation_loss=.3,p95_ms=1)]
        self.assertEqual(select_feasible(rows,2),'a')
        for budget in [-1,float('nan')]:
            with self.assertRaises(ValueError):select_feasible(rows,budget)
        for field in ['validation_loss','p95_ms']:
            invalid=[dict(rows[0],**{field:float('nan')})]
            with self.assertRaises(ValueError):select_feasible(invalid,2)

    def test_intersection_preserves_dataset_unit(self):
        panels={'random':['a/random','b/random','c/random'],'temporal':['c/temporal','a/temporal']}
        self.assertEqual(matched_datasets(panels),{'random':['a/random','c/random'],'temporal':['a/temporal','c/temporal']})
        with self.assertRaises(ValueError):matched_datasets({'random':['a/random'],'temporal':['b/temporal']})

if __name__=='__main__':unittest.main()

"""Behavioral tests: missingness, selection contamination and incompatible units."""
import copy,math,unittest
from relkit.multitask_l193 import choose_config,summarize_task,aggregate_suite
class Contracts(unittest.TestCase):
 def setUp(self):
  self.task=dict(id='a',group='relbench_classification',metric='AUROC',normalizer=None)
  self.rows=[dict(task='a',seed=s,status='COMPLETE',score=v) for s,v in enumerate([.6,.7,.8])]
 def test_selection_uses_full_validation_and_stable_ties(self):
  rows=[dict(id='b',split='val',score=.7),dict(id='a',split='val',score=.7)]
  self.assertEqual(choose_config(rows,['a','b'],'AUROC'),'a')
  rows[0]['score']=.9;self.assertEqual(choose_config(rows,['a','b'],'MAE'),'a')
 def test_selection_refuses_test_missing_duplicates_nonfinite(self):
  rows=[dict(id='a',split='val',score=.7),dict(id='b',split='val',score=.8)]
  for bad in [rows[:1],rows+[rows[0]], [dict(rows[0],split='test'),rows[1]],[dict(rows[0],score=math.nan),rows[1]]]:
   with self.assertRaises(ValueError):choose_config(bad,['a','b'],'AUROC')
 def test_seed_summary_and_variance(self):
  r=summarize_task(self.task,self.rows,[0,1,2]);self.assertAlmostEqual(r['mean'],.7);self.assertAlmostEqual(r['sample_sd'],.1);self.assertEqual(r['completed_seeds'],3)
 def test_missing_seed_is_explicit_not_a_mean(self):
  rows=copy.deepcopy(self.rows);rows[2].update(status='NOT_RUN',score=None)
  r=summarize_task(self.task,rows,[0,1,2]);self.assertIsNone(r['mean']);self.assertEqual(r['completed_seeds'],2)
  for bad in [rows[:2],rows+[rows[0]],[dict(rows[0],score=math.inf)]+rows[1:],[dict(rows[0],task='wrong')]+rows[1:]]:
   with self.assertRaises(ValueError):summarize_task(self.task,bad,[0,1,2])
 def test_suite_coverage_never_drops_tasks(self):
  other=dict(self.task,id='b');a=summarize_task(self.task,self.rows,[0,1,2]);b=summarize_task(other,[dict(r,task='b',status='NOT_RUN',score=None) for r in self.rows],[0,1,2])
  r=aggregate_suite([self.task,other],[a,b])['relbench_classification'];self.assertEqual(r['complete_tasks'],1);self.assertIsNone(r['mean']);self.assertEqual(r['expected_tasks'],2)
  with self.assertRaises(ValueError):aggregate_suite([self.task,other],[a])
 def test_regression_requires_known_normalizers(self):
  tasks=[dict(id=k,group='relbench_regression',metric='MAE',normalizer=n) for k,n in [('a',10),('b',.1)]]
  sums=[dict(task='a',status='COMPLETE',mean=5),dict(task='b',status='COMPLETE',mean=.2)]
  r=aggregate_suite(tasks,sums)['relbench_regression'];self.assertEqual(r['mean'],1.25)
  tasks[1]['normalizer']=None;self.assertIsNone(aggregate_suite(tasks,sums)['relbench_regression']['mean'])
 def test_auc_macro_equal_task_weight(self):
  tasks=[self.task,dict(self.task,id='b')];sums=[dict(task='a',status='COMPLETE',mean=.6),dict(task='b',status='COMPLETE',mean=.8)]
  self.assertAlmostEqual(aggregate_suite(tasks,sums)['relbench_classification']['mean'],.7)
if __name__=='__main__':unittest.main()

"""Behavioral acceptance checks: native sampling, independent forward oracle, time."""
import importlib.util,json,unittest,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class Contract(unittest.TestCase):
 def test_complete_contract(self):
  self.assertIsNotNone(importlib.util.find_spec('relkit.batching_l098'),'Missing L098 implementation')
  from relkit.batching_l098 import audit_graph, temporal_audit
  for seed in range(32):
   for size in [1,4,24]:
    report=audit_graph(seed,size)
    self.assertLess(report['max_logit_gap'],2e-6)
    self.assertLess(report['max_gradient_gap'],2e-6)
    self.assertEqual(report['seed_coverage'],list(range(24)))
  self.assertEqual(temporal_audit()['status'],'PASS')
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
 if r.wasSuccessful():(P/'_verify_l098_results.json').write_text(json.dumps({'status':'PASS','graphs':32,'batch_sizes':[1,4,24],'full_configurations':96,'temporal':'PASS'},indent=2)+'\n')
 sys.exit(not r.wasSuccessful())

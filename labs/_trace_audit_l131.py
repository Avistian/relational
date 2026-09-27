"""Independent arithmetic audit of every saved full-data forward/backward trace."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
reports=[]
for seed in range(5):
 t=json.loads((P/f'evidence/l131/paper/seed-{seed}/stack-trace.json').read_text());assert t['status']=='PASS' and t['batch_size']==512
 assert t['source_sha256']==hashlib.sha256((P/'relkit/stack_l131.py').read_bytes()).hexdigest()
 stages=t['stages']
 for kind,v in t['inputs'].items():assert stages['row_encoder'][kind]['shape']==[v['rows'],128]
 for kind,v in t['time_inputs'].items():
  days=[(v['cutoff_seconds'][q]-tm)/86400 for q,tm in zip(v['owner'],v['node_seconds'])]
  assert all(x>=0 for x in days)
  np.testing.assert_allclose(days,stages['relative_days'][kind]['first_rows'],rtol=1e-6,atol=1e-5)
  if v['node_seconds']:
   h=np.array(stages['row_encoder'][kind]['first_rows']);z=np.array(stages['time_encoder'][kind]['first_rows']);added=np.array(stages['time_added'][kind]['first_rows'])
   np.testing.assert_allclose(h+z,added,rtol=1e-5,atol=1e-6)
 for layer in [1,2]:
  for kind,v in stages[f'layer_{layer}_normalized'].items():
   np.testing.assert_allclose(np.maximum(np.array(v['first_rows']),0),stages[f'layer_{layer}_relu'][kind]['first_rows'])
 assert stages['root_readout']['drivers']['shape']==[512,128]
 assert stages['prediction']['drivers']['shape']==[512,1]
 loss=math.fsum(abs(x-y) for x,y in zip(t['predictions'],t['targets']))/512
 assert abs(loss-t['loss'])<2e-6
 assert t['gradients']['encoder']['nonfinite']==640
 assert t['parity']['nonfinite_gradient_parameters']==['encoder.encoders.results.encoder.encoder_dict.numerical.weight']
 assert t['parity']['nonfinite_masks']=='MATCH'
 assert t['parity']['max_output_error']<1e-5 and t['parity']['max_gradient_error']<1e-6
 assert t['detached_gradients']['encoder']['with_gradient']==0
 for name in ['time','gnn','head']:
  assert t['gradients'][name]['nonfinite']==0
  assert math.isclose(t['gradients'][name]['finite_l2'],t['detached_gradients'][name]['finite_l2'],rel_tol=1e-5,abs_tol=1e-6)
 reports.append(dict(seed=seed,rows=sum(x['rows'] for x in t['inputs'].values()),loss=t['loss'],parity=t['parity']))
r=dict(status='PASS',seeds=reports,nonfinite_gradient_entries_per_seed=640,scope='All saved first-minibatch shapes, age arithmetic, addition, ReLU, loss and gradient reports independently checked; full tensors are summarized, not retained',raw_gradient_health='NONFINITE_IN_RELEASED_AND_INSTRUMENTED',finite_gradient_parity='PASS')
(P/'_trace_audit_l131_results.json').write_text(json.dumps(r,indent=2));print(r)

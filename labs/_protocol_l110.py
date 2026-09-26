"""Full-stream independent boundary census and release trainer regression."""
import ast,json,tempfile
from pathlib import Path
import numpy as np
import torch
from relkit import checkpoint_l110 as c,tgn_l102 as old
P=Path(__file__).resolve().parent;torch.set_num_threads(1)
nodes,edges,data,audit=c.load_wikipedia(P/'data/l102');counts={}
for key,events in data.items():
 if key=='full':continue
 parts=list(c.strict_batches(events));release=list(c.batches(events));cross=lambda xs:int(sum(a['t'][-1]==b['t'][0] for a,b in zip(xs,xs[1:])))
 assert cross(parts)==0
 assert all(np.all(c.legal_history(a['t'],a['t'],b['t'][0])) for a,b in zip(parts,parts[1:]))
 np.testing.assert_array_equal(np.concatenate([b['e'] for b in parts]),events['e'])
 counts[key]={'events':len(events['e']),'release_batches':len(release),'clean_batches':len(parts),'release_tied_boundaries':cross(release),'clean_tied_boundaries':cross(parts),'largest_clean_batch':max(len(x['e']) for x in parts)}
# Model and preprocessing remain the previously source-audited implementation.
def defs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
a=defs(P/'relkit/tgn_l102.py');b=defs(P/'relkit/checkpoint_l110.py');same=[k for k in a if k not in ['run_training','evaluate']]
assert all(a[k]==b[k] for k in same)
small={k:({f:x[:(400 if k=='train' else 200)] for f,x in ev.items()} if k!='full' else ev) for k,ev in data.items()}
with tempfile.TemporaryDirectory() as tmp:
 r=old.run_training(nodes,edges,small,seed=19,epochs=2,output=Path(tmp)/'old')
 q=c.run_training(nodes,edges,small,seed=19,epochs=2,output=Path(tmp)/'new')
 for name in ['test','new_test','selected_epoch']:assert r[name]==q[name]
 x=np.load(Path(tmp)/'old/seed-19-predictions.npz');y=np.load(Path(tmp)/'new/seed-19-predictions.npz')
 for key in x.files:np.testing.assert_array_equal(x[key],y[key])
 # Forced early-stop case verifies the different restore branch in a real trainer.
 z=c.run_training(nodes,edges,small,seed=19,epochs=2,patience=0,output=Path(tmp)/'clean',arm='clean')
 assert z['selected_epoch']==0 and z['early_stopped']
report={'status':'PASS','counts':counts,'unchanged_definitions':same,'release_training_regression':'EXACT probabilities and metrics, 2 epochs seed19','clean_early_stop':'PASS','availability_assumption':'observed_time=event_time; real ingestion unavailable'}
(P/'_protocol_l110_results.json').write_text(json.dumps(report,indent=2));print(report)

"""Admit the complete remaining grid only from authenticated full pilot predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
from relkit.composite_l182 import keyed_auc
P=Path(__file__).resolve().parent;E=P/'evidence/b23';data=np.load(P/'evidence/l166/prepared.npz')
r=json.loads((E/'remote-pilot.json').read_text());assert len(r['records'])==3
for row in r['records']:
 path=E/'pilot-1'/f"{row['arm']}-0.npz"
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 x=np.load(path);np.testing.assert_array_equal(x['keys'],data['test_keys']);np.testing.assert_array_equal(x['label'],data['y_test']);np.testing.assert_array_equal(x['support_keys'],data['train_keys'][data['support'][0]])
 assert abs(keyed_auc(x['keys'],x['label'],x['keys'],x['probability'])-row['auc'])<1e-12
seconds=json.loads((E/'pilot-1/cost.json').read_text())['worker_body_seconds']
projected=seconds*10+120 # conservative: repeat all import/loading overhead ten times.
assert projected<5300 and (projected+630)*.00028372+2<8
out=dict(decision='PROCEED',pilot_body_seconds=seconds,projected_remaining_upper_seconds=projected,remaining_reservation_seconds=5330,total_reservation_usd=5960*.00028372+2,scope='All remaining27evaluations; no model or seed reduction')
(E/'admission.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

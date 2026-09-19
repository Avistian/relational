"""Independent reconstruction of saved prediction metrics and execution coverage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();out=Path(args.run)
r=json.loads((out/'result.json').read_text());z=np.load(out/'test_predictions.npz')
assert hashlib.sha256((out/'test_predictions.npz').read_bytes()).hexdigest()==r['prediction_sha256']
assert len(z['ids'])==5524 and len(np.unique(z['ids']))==5524
assert np.all(np.isfinite(z['logits']))
f1=f1_score(z['labels'],z['logits']>0,average='micro',zero_division=0)
loss=np.mean(np.logaddexp(0,z['logits'].astype(np.float64))-z['labels']*z['logits'])
assert abs(f1-r['test']['micro_f1'])<1e-12
assert abs(loss-r['test']['loss'])<1e-6
parts=np.load(out/'partitions.npz');ids=parts['train_ids'];joined=np.concatenate([parts[k] for k in parts.files if k.startswith('p')])
assert len(ids)==44906 and len(np.unique(ids))==44906
np.testing.assert_array_equal(np.sort(joined),np.arange(len(ids)))
assert not np.intersect1d(ids,z['ids']).size
# Check identity and labels against the hash-verified source, not just the saved arrays.
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from cluster_gcn_l089 import load_ppi
d=load_ppi(Path(__file__).resolve().parent/'data/l089')
np.testing.assert_array_equal(ids,np.flatnonzero(d['masks']['train']))
np.testing.assert_array_equal(z['ids'],np.flatnonzero(d['masks']['test']))
np.testing.assert_array_equal(z['labels'],d['y'][z['ids']])
assert len(r['curves'])==r['config']['epochs'] and r['selection']=='final_epoch'
full=all(r['config'].get(k)==v for k,v in {'hidden':2048,'layers':5,'epochs':400,'num_parts':50,'q':1,'dropout':.2,'lr':.01,'partition_method':'metis','partition_seed':1}.items())
report={'status':'PASS','run':str(out),'recomputed_micro_f1':f1,'recomputed_bce':float(loss),'full_recipe_executed':full,'historical_parity':'INCOMPARABLE'}
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

"""Read validation arrays only; persist nomination before any test slice analysis."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
from relkit.error_reg_l137 import paired_errors,nominate_slice
P=Path(__file__).resolve().parent;E=P/'evidence/l137';out=E/'frozen.json'
assert not out.exists(),'Do not reselect after seeing test results'
a=np.load(E/'diagnostics.npz');masks={k[len('val_mask_'):]:a[k] for k in a.files if k.startswith('val_mask_')};keys=list(zip(a['val_entity'],a['val_time']));deltas=[];hashes={}
for seed in range(5):
    paths=[E/f'final/lr005-full/seed-{seed}/predictions.npz',E/f'fe/paper/seed-{seed}/predictions.npz']
    g,f=[np.load(p) for p in paths]
    np.testing.assert_allclose(g['val_target'],f['val_target'],rtol=0,atol=1e-6)
    deltas.append(paired_errors(keys,f['val_target'],list(zip(g['val_entity'],g['val_time'])),g['val_pred'],list(zip(f['val_entity'],f['val_time'])),f['val_pred']))
    for p in paths:hashes[str(p.relative_to(P.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
r=nominate_slice(np.mean(deltas,axis=0),a['val_entity'],masks)
r.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=hashes,diagnostics_sha256=hashlib.sha256((E/'diagnostics.npz').read_bytes()).hexdigest(),analysis_source_sha256=hashlib.sha256((P/'relkit/error_reg_l137.py').read_bytes()).hexdigest(),test_arrays_read=False,holdout_boundary='Split reused in prior lessons; not a new pristine confirmatory holdout. This operator only nominates using validation.')
out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))

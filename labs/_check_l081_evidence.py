"""Replay stored checkpoints and test that held-out labels cannot change training."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from relkit.mpnn_l081 import SparseGGNN
from relkit.qm9_l081 import load_qm9,split_ids,predict,fit_trial,sha256
LAB=Path(__file__).resolve().parent
torch.set_num_threads(1)
graphs,data=load_qm9(LAB/'data/cache/l081',512)
errors=[]
for seed in [81,82,83]:
    folder=LAB/'evidence/l081'/str(seed);r=json.loads((folder/'result.json').read_text())
    assert data==r['data']
    assert sha256(folder/'split.json')==r['split_sha256']
    assert sha256(folder/'predictions.npz')==r['prediction_sha256']
    split=split_ids(len(graphs),seed)
    stored=json.loads((folder/'split.json').read_text())
    for name,ids in split.items():assert stored[name]==[int(graphs[i].molecule_id) for i in ids]
    assert not (set(stored['train']) & set(stored['test']))
    assert not (set(stored['train']) & set(stored['validation']))
    assert not (set(stored['validation']) & set(stored['test']))
    checkpoint=torch.load(folder/'selected.pt',weights_only=False,map_location='cpu');record=checkpoint['record']
    model=SparseGGNN(width=record['config']['width'],steps=record['config']['rounds']);model.load_state_dict(checkpoint['state_dict'])
    fresh=predict(model,graphs,split['test'],record['mean'],record['std'],'cpu')
    z=np.load(folder/'predictions.npz');np.testing.assert_allclose(fresh,z['prediction'],atol=0,rtol=0)
    assert abs(np.mean(np.abs(fresh-z['target']))-r['test_mae_debye'])<1e-12
    errors.append(float(np.max(np.abs(fresh-z['prediction']))))
# Poison TEST labels only: neither selected state nor validation record can change.
small=graphs[:40];split=split_ids(40,81);poison=copy.deepcopy(small)
for i in split['test']:poison[i].y+=10000
config={'width':16,'rounds':3,'lr':.001,'decay_start':.5,'final_factor':.1,'updates':3,'validate_every':1}
a,ra=fit_trial(small,split,config,81);b,rb=fit_trial(poison,split,config,81)
assert ra==rb
for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
report={'status':'PASS','checkpoint_prediction_max_errors':errors,'test_label_poisoning':'selected state and validation history unchanged','data_hashes_enforced':True,'split_disjointness':True}
(LAB/'_check_l081_evidence_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

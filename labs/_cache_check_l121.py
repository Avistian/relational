"""Cached deterministic inputs must preserve tensors, outputs and gradients."""
import copy,json
from pathlib import Path
import torch
from relkit.cvitkovic_l118 import PreparedGraphs,collate_graphs,CvitkovicGCN,feature_schema
from relkit.cache_l121 import cache_record,collate_cached
P=Path(__file__).resolve().parent
info=json.loads((P/'sources/l118/data/homecreditdefaultrisk/homecreditdefaultrisk.db_info.json').read_text())
ids=json.loads((P/'data/l118/pilot/ids.json').read_text())[:8];ds=PreparedGraphs(P/'data/l118/pilot',ids);records=[ds[i] for i in range(len(ds))]
a,y,ii=collate_graphs(records,info);b,z,jj=collate_cached([cache_record(r,info) for r in records])
for key in a[0]:
 for x1,x2 in zip(a[0][key],b[0][key]):assert torch.equal(x1,x2),key
for x1,x2 in zip(a[1:4],b[1:4]):assert torch.equal(x1,x2)
assert a[4]==b[4] and torch.equal(y,z) and (ii==jj).all()
torch.set_num_threads(2);torch.manual_seed(12);m=CvitkovicGCN(feature_schema(info),info['node_type_to_int']);n=copy.deepcopy(m)
torch.manual_seed(99);p=m(*a);p.sum().backward();torch.manual_seed(99);q=n(*b);q.sum().backward()
assert torch.equal(p,q)
for p1,p2 in zip(m.parameters(),n.parameters()):assert torch.equal(p1.grad,p2.grad)
r={'status':'PASS','graphs':len(ids),'encoded_tensors':'EXACT','training_outputs':'EXACT','gradients':'EXACT','dropout':'same RNG state, stochastic model still executed','boundary':'controlled eight-graph batch; no training score'}
(P/'_cache_check_l121_results.json').write_text(json.dumps(r,indent=2));print(r)
